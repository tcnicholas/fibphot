from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from ..state import PhotometryState
from .peak_aligned import PeakTriggeredAverage, result_to_event_aligned
from .report import AnalysisResult, AnalysisWindow
from .statistics import nan_sem


def _window_to_mask(t: np.ndarray, window: AnalysisWindow | None) -> np.ndarray:
    if window is None:
        return np.isfinite(t)
    if window.ref != "seconds":
        i0 = max(0, int(window.start))
        i1 = min(t.size, int(window.end))
        m = np.zeros_like(t, dtype=bool)
        m[i0:i1] = True
        return m & np.isfinite(t)
    lo = float(min(window.start, window.end))
    hi = float(max(window.start, window.end))
    return np.isfinite(t) & (t >= lo) & (t <= hi)


def _safe_detrend(x: np.ndarray) -> np.ndarray:
    from scipy.signal import detrend

    arr = np.asarray(x, dtype=float)
    if arr.size < 3 or np.sum(np.isfinite(arr)) < 3:
        return arr - np.nanmean(arr)
    good = np.isfinite(arr)
    if not np.all(good):
        idx = np.arange(arr.size, dtype=float)
        arr = arr.copy()
        arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    return detrend(arr)


def cross_correlation(
    x: Sequence[float],
    y: Sequence[float],
    *,
    dt: float,
    max_lag_s: float | None = None,
    detrend: bool = True,
    normalise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import correlate, correlation_lags

    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    a = a[m]
    b = b[m]
    if detrend:
        a = _safe_detrend(a)
        b = _safe_detrend(b)
    else:
        a = a - np.nanmean(a)
        b = b - np.nanmean(b)
    corr = correlate(a, b, mode="full")
    if normalise:
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom > 0:
            corr = corr / denom
    lags = correlation_lags(a.size, b.size, mode="full") * float(dt)
    if max_lag_s is not None:
        keep = np.abs(lags) <= float(max_lag_s)
        lags = lags[keep]
        corr = corr[keep]
    return lags.astype(float), corr.astype(float)


def _fdr_bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    vals = p[finite]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    n = ranked.size
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[finite] = tmp
    return out


def _empty_granger_result() -> dict[str, np.ndarray]:
    empty = np.array([], dtype=float)
    return {
        "lag_s": empty,
        "p_x_to_y": empty,
        "p_y_to_x": empty,
        "p_x_to_y_fdr": empty,
        "p_y_to_x_fdr": empty,
    }


def _run_granger_once(data: np.ndarray, maxlag: int) -> list[float]:
    """Run statsmodels Granger once and collect p-values up to ``maxlag``.

    ``statsmodels.tsa.stattools.grangercausalitytests`` already computes every
    lag from 1..maxlag. This helper calls it once at the largest feasible lag,
    then extracts all lower-lag p-values.  Statsmodels can reject a requested
    lag because the downsampled trace is too short, has near-constant columns,
    or is numerically singular.  For GUI use, that should not silently produce
    an empty-looking plot: we progressively reduce the requested lag and pad any
    untestable high-lag values with NaNs.
    """

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception as exc:
        raise ImportError(
            "Granger causality requires statsmodels. Install `statsmodels`."
        ) from exc

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 8:
        return [np.nan] * int(maxlag)
    if not np.all(np.isfinite(arr)):
        good = np.all(np.isfinite(arr), axis=1)
        arr = arr[good]
    if arr.shape[0] < 8 or np.any(np.nanstd(arr, axis=0) <= 1e-12):
        return [np.nan] * int(maxlag)

    # Standardise for numerical stability.  This does not change Granger-test
    # p-values, but it improves conditioning for small/interactive windows.
    arr = (arr - np.nanmean(arr, axis=0)) / np.nanstd(arr, axis=0)

    import warnings

    requested = int(max(1, maxlag))
    last_error: Exception | None = None
    for lag_max in range(requested, 0, -1):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="verbose is deprecated",
                    category=FutureWarning,
                )
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    results = grangercausalitytests(
                        arr, maxlag=lag_max, verbose=False
                    )
                except (
                    TypeError
                ):  # statsmodels versions where ``verbose`` has changed.
                    results = grangercausalitytests(arr, maxlag=lag_max)
            p_values: list[float] = []
            for lag in range(1, lag_max + 1):
                try:
                    p_values.append(float(results[lag][0]["ssr_ftest"][1]))
                except Exception:
                    p_values.append(np.nan)
            if lag_max < requested:
                p_values.extend([np.nan] * (requested - lag_max))
            return p_values
        except (
            Exception
        ) as exc:  # progressively reduce lag rather than failing the GUI plot.
            last_error = exc
            continue

    return [np.nan] * requested


def granger_pvalues(
    x: Sequence[float],
    y: Sequence[float],
    *,
    dt: float,
    max_lag_s: float = 10.0,
    target_dt: float | None = 1.0,
    max_lag_steps: int | None = 20,
    max_samples: int | None = 2000,
    detrend: bool = True,
    difference: bool = True,
    fdr: bool = True,
) -> dict[str, np.ndarray]:
    """Return pairwise Granger-causality p-values in both directions.

    The function is deliberately conservative for interactive use. Long
    photometry traces at acquisition rate can make VAR/Granger tests extremely
    slow. The data are therefore optionally downsampled to ``target_dt``, capped
    to ``max_samples``, and the number of tested lags is capped by
    ``max_lag_steps``. Set these caps to ``None`` only for offline analyses.
    """

    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 12:
        return _empty_granger_result()

    a = a[m]
    b = b[m]
    if detrend:
        a = _safe_detrend(a)
        b = _safe_detrend(b)

    dt2 = float(dt)
    if target_dt is not None and float(target_dt) > dt2:
        factor = max(1, int(round(float(target_dt) / dt2)))
        a = a[::factor]
        b = b[::factor]
        dt2 *= factor

    if (
        max_samples is not None
        and max_samples > 0
        and a.size > int(max_samples)
    ):
        factor = int(np.ceil(a.size / int(max_samples)))
        a = a[::factor]
        b = b[::factor]
        dt2 *= factor

    if difference:
        a = np.diff(a)
        b = np.diff(b)

    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    if n < 12:
        return _empty_granger_result()

    maxlag_time = (
        int(np.floor(float(max_lag_s) / dt2))
        if max_lag_s is not None
        else n // 3
    )
    maxlag_possible = max(1, min(n // 3, n - 5))
    maxlag = min(maxlag_time, maxlag_possible)
    if max_lag_steps is not None and int(max_lag_steps) > 0:
        maxlag = min(maxlag, int(max_lag_steps))
    if maxlag < 1:
        return _empty_granger_result()

    # statsmodels convention: second column Granger-causes first column.
    try:
        p_xy = _run_granger_once(np.column_stack([b, a]), maxlag)
    except Exception:
        p_xy = [np.nan] * maxlag
    try:
        p_yx = _run_granger_once(np.column_stack([a, b]), maxlag)
    except Exception:
        p_yx = [np.nan] * maxlag

    pxy = np.asarray(p_xy, dtype=float)
    pyx = np.asarray(p_yx, dtype=float)
    lags = np.arange(1, maxlag + 1, dtype=float) * dt2
    return {
        "lag_s": lags,
        "p_x_to_y": pxy,
        "p_y_to_x": pyx,
        "p_x_to_y_fdr": _fdr_bh(pxy) if fdr else pxy.copy(),
        "p_y_to_x_fdr": _fdr_bh(pyx) if fdr else pyx.copy(),
    }


@dataclass(frozen=True, slots=True)
class CrossCorrelationAnalysis:
    x_signal: str
    y_signal: str
    window: AnalysisWindow | None = None
    max_lag_s: float = 10.0
    detrend: bool = True
    normalise: bool = True
    name: str = "cross_correlation"

    def __call__(self, state: PhotometryState) -> AnalysisResult:
        t = np.asarray(state.time_seconds, dtype=float)
        m = _window_to_mask(t, self.window)
        x = np.asarray(state.channel(self.x_signal), dtype=float)[m]
        y = np.asarray(state.channel(self.y_signal), dtype=float)[m]
        tt = t[m]
        dt = (
            float(np.nanmedian(np.diff(tt)))
            if tt.size >= 2
            else 1.0 / state.sampling_rate
        )
        lags, corr = cross_correlation(
            x,
            y,
            dt=dt,
            max_lag_s=self.max_lag_s,
            detrend=self.detrend,
            normalise=self.normalise,
        )
        if corr.size:
            imax = int(np.nanargmax(np.abs(corr)))
            metrics = {
                "max_abs_corr": float(corr[imax]),
                "lag_at_max_abs_corr_s": float(lags[imax]),
            }
        else:
            metrics = {"max_abs_corr": np.nan, "lag_at_max_abs_corr_s": np.nan}
        return AnalysisResult(
            self.name,
            f"{self.x_signal},{self.y_signal}",
            self.window,
            asdict(self),
            metrics,
            {"lag_s": lags, "correlation": corr},
        )


@dataclass(frozen=True, slots=True)
class GrangerCausalityAnalysis:
    x_signal: str
    y_signal: str
    window: AnalysisWindow | None = None
    max_lag_s: float = 10.0
    target_dt: float | None = 1.0
    max_lag_steps: int | None = 20
    max_samples: int | None = 2000
    detrend: bool = True
    difference: bool = True
    fdr: bool = True
    name: str = "granger_causality"

    def __call__(self, state: PhotometryState) -> AnalysisResult:
        t = np.asarray(state.time_seconds, dtype=float)
        m = _window_to_mask(t, self.window)
        x = np.asarray(state.channel(self.x_signal), dtype=float)[m]
        y = np.asarray(state.channel(self.y_signal), dtype=float)[m]
        tt = t[m]
        dt = (
            float(np.nanmedian(np.diff(tt)))
            if tt.size >= 2
            else 1.0 / state.sampling_rate
        )
        out = granger_pvalues(
            x,
            y,
            dt=dt,
            max_lag_s=self.max_lag_s,
            target_dt=self.target_dt,
            max_lag_steps=self.max_lag_steps,
            max_samples=self.max_samples,
            detrend=self.detrend,
            difference=self.difference,
            fdr=self.fdr,
        )
        metrics = {
            "min_p_x_to_y": float(np.nanmin(out["p_x_to_y"]))
            if out["p_x_to_y"].size
            else np.nan,
            "min_p_y_to_x": float(np.nanmin(out["p_y_to_x"]))
            if out["p_y_to_x"].size
            else np.nan,
            "n_lags_tested": float(out["lag_s"].size),
            "effective_max_lag_s": float(out["lag_s"][-1])
            if out["lag_s"].size
            else np.nan,
        }
        return AnalysisResult(
            self.name,
            f"{self.x_signal},{self.y_signal}",
            self.window,
            asdict(self),
            metrics,
            out,
            "Granger p-values; lower values suggest predictive information, not mechanistic causation.",
        )


def _normalise_channels_for_state(
    channels: Sequence[str] | str | None,
    state: PhotometryState,
    *,
    include: Sequence[str | None] = (),
) -> tuple[str, ...]:
    """Return valid channel names, preserving state order where possible."""
    if channels is None or channels == "all":
        requested: list[str] = list(state.channel_names)
    elif isinstance(channels, str):
        requested = [
            c.strip().lower() for c in channels.split(",") if c.strip()
        ]
    else:
        requested = [str(c).strip().lower() for c in channels if str(c).strip()]

    for name in include:
        if name is not None and str(name).strip():
            key = str(name).strip().lower()
            if key not in requested:
                requested.append(key)

    valid: list[str] = []
    for name in requested:
        try:
            idx = state.idx(name)
        except Exception:
            raise KeyError(
                f"Unknown channel {name!r}. Available: {state.channel_names}"
            ) from None
        actual = state.channel_names[idx]
        if actual not in valid:
            valid.append(actual)
    return tuple(valid)


def _parse_pairs(
    pairs: Sequence[Sequence[str]] | Sequence[tuple[str, str]] | str | None,
) -> tuple[tuple[str, str], ...]:
    """Parse connectivity pairs from a Python object or GUI text.

    Accepted text examples: ``"gcamp,rgeco"`` or
    ``"gcamp:rgeco; gcamp:iso"``.
    """
    if pairs is None or pairs == "":
        return ()
    if isinstance(pairs, str):
        chunks = [
            c.strip()
            for c in pairs.replace(";", "\n").splitlines()
            if c.strip()
        ]
        out: list[tuple[str, str]] = []
        for chunk in chunks:
            sep = ":" if ":" in chunk else ","
            bits = [b.strip().lower() for b in chunk.split(sep) if b.strip()]
            if len(bits) != 2:
                raise ValueError(
                    "connectivity_pairs text must contain pairs like "
                    "'gcamp:rgeco' or 'gcamp,rgeco'."
                )
            out.append((bits[0], bits[1]))
        return tuple(out)
    out = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(
                "Each connectivity pair must contain exactly two channel names."
            )
        out.append((str(pair[0]).strip().lower(), str(pair[1]).strip().lower()))
    return tuple(out)


def _default_event_pairs(
    event_signal: str, channels: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    event = event_signal.lower()
    pairs = []
    for channel in channels:
        c = channel.lower()
        if c != event:
            pairs.append((event, c))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class PeakAlignedConnectivityAnalysis:
    """Align all requested signals to peaks in one event signal, then analyse them.

    This is intended for questions such as: "find peaks in green, align all
    epochs to those green-peak times, then inspect the average green response
    and the concurrent red/isosbestic response". Connectivity curves are then
    computed from the same aligned epoch stack.

    Parameters
    ----------
    event_signal
        Channel used for event/peak detection and time alignment.
    channels
        Channels to extract around each event. Use ``"all"`` to include all
        processed channels, or a comma-separated/list of names. ``event_signal``
        is always included.
    connectivity_pairs
        Optional channel pairs for cross-correlation/Granger analysis. If not
        supplied, the default is ``event_signal`` paired with every other
        aligned channel.
    x_signal, y_signal
        Convenience pair fields for simple two-signal workflows. These are
        folded into ``connectivity_pairs`` and ``channels`` when supplied.
    """

    event_signal: str
    channels: Sequence[str] | str | None = "all"
    connectivity_pairs: Sequence[Sequence[str]] | str | None = None
    x_signal: str | None = None
    y_signal: str | None = None
    detector: Literal["peaks", "template"] = "template"
    detector_params: Mapping[str, Any] | None = None
    window: AnalysisWindow | None = None
    t_before_s: float = 20.0
    t_after_s: float = 20.0
    target_fs: float | None = None
    dt: float | None = None
    baseline_window_s: tuple[float, float] | None = None
    max_lag_s: float = 10.0
    detrend: bool = True
    normalise: bool = True
    run_granger: bool = False
    granger_mode: Literal["mean_epoch", "per_event"] = "mean_epoch"
    granger_target_dt: float | None = 1.0
    granger_max_lag_steps: int | None = 10
    granger_max_samples: int | None = 1000
    granger_max_events: int | None = 50
    name: str = "peak_aligned_connectivity"

    def __call__(self, state: PhotometryState) -> AnalysisResult:
        include = [self.event_signal, self.x_signal, self.y_signal]
        channels = _normalise_channels_for_state(
            self.channels, state, include=include
        )

        avg = PeakTriggeredAverage(
            event_signal=self.event_signal,
            channels=channels,
            detector=self.detector,
            detector_params=self.detector_params,
            window=self.window,
            t_before_s=self.t_before_s,
            t_after_s=self.t_after_s,
            target_fs=self.target_fs,
            dt=self.dt,
            baseline_window_s=self.baseline_window_s,
        )(state)
        if not avg.arrays:
            return AnalysisResult(
                self.name,
                self.event_signal,
                self.window,
                self._params(channels, ()),
                {"n_events": 0.0, "n_channels": float(len(channels))},
                {},
                "No aligned events available.",
            )

        aligned = result_to_event_aligned(avg)
        channel_names = tuple(str(c).lower() for c in aligned.channel_names)
        name_to_i = {name: i for i, name in enumerate(channel_names)}
        dt_epoch = float(np.nanmedian(np.diff(aligned.time_relative_s)))

        pairs = list(_parse_pairs(self.connectivity_pairs))
        if self.x_signal and self.y_signal:
            pair = (self.x_signal.lower(), self.y_signal.lower())
            if pair not in pairs:
                pairs.insert(0, pair)
        if not pairs:
            pairs = list(_default_event_pairs(self.event_signal, channel_names))
        # Keep only valid, non-duplicate pairs.
        valid_pairs: list[tuple[str, str]] = []
        for a, b in pairs:
            aa, bb = a.lower(), b.lower()
            if aa not in name_to_i or bb not in name_to_i:
                raise KeyError(
                    f"Connectivity pair ({a!r}, {b!r}) is not present in aligned channels {channel_names}."
                )
            pair = (aa, bb)
            if pair not in valid_pairs:
                valid_pairs.append(pair)

        arrays: dict[str, Any] = dict(avg.arrays)
        arrays.update(
            {
                "time_relative_s": aligned.time_relative_s,
                "event_time_s": aligned.event_times_s,
                "aligned_channel_names": np.asarray(
                    aligned.channel_names, dtype=str
                ),
                "aligned_traces": aligned.data,
                "alignment_event_signal": np.asarray(
                    [self.event_signal.lower()], dtype=str
                ),
                "connectivity_pair_labels": np.asarray(
                    [f"{a}->{b}" for a, b in valid_pairs], dtype=str
                ),
            }
        )
        metrics: dict[str, Any] = {
            "n_events": float(aligned.n_events),
            "n_channels": float(len(aligned.channel_names)),
            "n_connectivity_pairs": float(len(valid_pairs)),
        }

        lag_ref: np.ndarray | None = None
        corr_by_pair: list[np.ndarray] = []
        corr_pair_event_times: list[np.ndarray] = []
        for a, b in valid_pairs:
            ix, iy = name_to_i[a], name_to_i[b]
            corrs: list[np.ndarray] = []
            kept_event_times: list[float] = []
            local_lag_ref: np.ndarray | None = None
            for ei in range(aligned.n_events):
                lags, corr = cross_correlation(
                    aligned.data[ei, ix],
                    aligned.data[ei, iy],
                    dt=dt_epoch,
                    max_lag_s=self.max_lag_s,
                    detrend=self.detrend,
                    normalise=self.normalise,
                )
                if corr.size == 0:
                    continue
                if local_lag_ref is None:
                    local_lag_ref = lags
                if corr.shape == local_lag_ref.shape:
                    corrs.append(corr)
                    kept_event_times.append(float(aligned.event_times_s[ei]))
            if local_lag_ref is None or not corrs:
                continue
            if lag_ref is None:
                lag_ref = local_lag_ref
            if local_lag_ref.shape != lag_ref.shape or not np.allclose(
                local_lag_ref, lag_ref
            ):
                continue
            c = np.vstack(corrs)
            corr_by_pair.append(c)
            corr_pair_event_times.append(
                np.asarray(kept_event_times, dtype=float)
            )
            mean = np.nanmean(c, axis=0)
            imax = int(np.nanargmax(np.abs(mean)))
            metrics[f"{a}_to_{b}_n_valid_correlations"] = float(c.shape[0])
            metrics[f"{a}_to_{b}_max_abs_corr_mean"] = float(mean[imax])
            metrics[f"{a}_to_{b}_lag_at_max_abs_corr_s"] = float(lag_ref[imax])

        if lag_ref is not None and corr_by_pair:
            # Shape: n_pairs x n_events_for_pair x n_lag. Different pairs may
            # have different valid event counts, so pad to a common length.
            n_pairs = len(corr_by_pair)
            max_events = max(c.shape[0] for c in corr_by_pair)
            n_lag = lag_ref.size
            padded = np.full((n_pairs, max_events, n_lag), np.nan, dtype=float)
            padded_event_times = np.full(
                (n_pairs, max_events), np.nan, dtype=float
            )
            for pi, c in enumerate(corr_by_pair):
                padded[pi, : c.shape[0], :] = c
                padded_event_times[pi, : corr_pair_event_times[pi].size] = (
                    corr_pair_event_times[pi]
                )
            arrays.update(
                {
                    "lag_s": lag_ref,
                    "corr_by_pair_event": padded,
                    "corr_pair_event_time_s": padded_event_times,
                    "corr_mean_by_pair": np.nanmean(padded, axis=1),
                    "corr_std_by_pair": np.nanstd(padded, axis=1, ddof=1),
                    "corr_sem_by_pair": nan_sem(padded, axis=1),
                    "corr_n_by_pair": np.sum(np.isfinite(padded), axis=1),
                }
            )
            # Preserve old/simple one-pair field names as a convenience for
            # table export and older plotting paths.
            arrays.update(
                {
                    "corr_mean": arrays["corr_mean_by_pair"][0],
                    "corr_std": arrays["corr_std_by_pair"][0],
                    "corr_sem": arrays["corr_sem_by_pair"][0],
                    "corr_n": arrays["corr_n_by_pair"][0],
                    "corr_by_event": padded[0],
                }
            )
            metrics.update(
                {
                    "n_valid_correlations": float(
                        np.sum(np.isfinite(padded[0, :, 0]))
                    ),
                    "max_abs_corr_mean": metrics.get(
                        f"{valid_pairs[0][0]}_to_{valid_pairs[0][1]}_max_abs_corr_mean",
                        np.nan,
                    ),
                    "lag_at_max_abs_corr_s": metrics.get(
                        f"{valid_pairs[0][0]}_to_{valid_pairs[0][1]}_lag_at_max_abs_corr_s",
                        np.nan,
                    ),
                }
            )

        if self.run_granger and aligned.n_events and valid_pairs:
            granger_lag_ref: np.ndarray | None = None
            pxy_by_pair: list[np.ndarray] = []
            pyx_by_pair: list[np.ndarray] = []

            if self.granger_mode == "mean_epoch":
                # Fast interactive default: test the mean event-aligned traces for
                # each pair once. This answers whether the average peri-event
                # waveform in one channel improves prediction of the other.
                mean_traces = np.nanmean(aligned.data, axis=0)
                for a, b in valid_pairs:
                    ix, iy = name_to_i[a], name_to_i[b]
                    out = granger_pvalues(
                        mean_traces[ix],
                        mean_traces[iy],
                        dt=dt_epoch,
                        max_lag_s=self.max_lag_s,
                        target_dt=self.granger_target_dt,
                        max_lag_steps=self.granger_max_lag_steps,
                        max_samples=self.granger_max_samples,
                        detrend=self.detrend,
                        difference=True,
                        fdr=True,
                    )
                    if out["lag_s"].size == 0:
                        continue
                    if granger_lag_ref is None:
                        granger_lag_ref = out["lag_s"]
                    if out[
                        "lag_s"
                    ].shape != granger_lag_ref.shape or not np.allclose(
                        out["lag_s"], granger_lag_ref
                    ):
                        continue
                    pxy_by_pair.append(out["p_x_to_y"][None, :])
                    pyx_by_pair.append(out["p_y_to_x"][None, :])

            elif self.granger_mode == "per_event":
                # Slower diagnostic mode. Limit event count by default so the
                # GUI remains responsive. Use a script with larger limits for
                # heavy offline analyses.
                event_indices = np.arange(aligned.n_events)
                if (
                    self.granger_max_events is not None
                    and aligned.n_events > int(self.granger_max_events)
                ):
                    event_indices = event_indices[
                        : int(self.granger_max_events)
                    ]

                for a, b in valid_pairs:
                    ix, iy = name_to_i[a], name_to_i[b]
                    pxy: list[np.ndarray] = []
                    pyx: list[np.ndarray] = []
                    local_lag_ref: np.ndarray | None = None
                    for ei in event_indices:
                        out = granger_pvalues(
                            aligned.data[int(ei), ix],
                            aligned.data[int(ei), iy],
                            dt=dt_epoch,
                            max_lag_s=self.max_lag_s,
                            target_dt=self.granger_target_dt,
                            max_lag_steps=self.granger_max_lag_steps,
                            max_samples=self.granger_max_samples,
                            detrend=self.detrend,
                            difference=True,
                            fdr=True,
                        )
                        if out["lag_s"].size == 0:
                            continue
                        if local_lag_ref is None:
                            local_lag_ref = out["lag_s"]
                        if out["lag_s"].shape == local_lag_ref.shape:
                            pxy.append(out["p_x_to_y"])
                            pyx.append(out["p_y_to_x"])
                    if local_lag_ref is None or not pxy:
                        continue
                    if granger_lag_ref is None:
                        granger_lag_ref = local_lag_ref
                    if (
                        local_lag_ref.shape != granger_lag_ref.shape
                        or not np.allclose(local_lag_ref, granger_lag_ref)
                    ):
                        continue
                    pxy_by_pair.append(np.vstack(pxy))
                    pyx_by_pair.append(np.vstack(pyx))

            else:
                raise ValueError(
                    "granger_mode must be 'mean_epoch' or 'per_event'."
                )

            if granger_lag_ref is not None and pxy_by_pair:
                max_events = max(a.shape[0] for a in pxy_by_pair)
                n_pairs = len(pxy_by_pair)
                n_lag = granger_lag_ref.size
                pxy_pad = np.full(
                    (n_pairs, max_events, n_lag), np.nan, dtype=float
                )
                pyx_pad = np.full(
                    (n_pairs, max_events, n_lag), np.nan, dtype=float
                )
                for pi, arr in enumerate(pxy_by_pair):
                    pxy_pad[pi, : arr.shape[0], :] = arr
                    pyx_pad[pi, : pyx_by_pair[pi].shape[0], :] = pyx_by_pair[pi]
                arrays.update(
                    {
                        "granger_lag_s": granger_lag_ref,
                        "p_x_to_y_mean_by_pair": np.nanmean(pxy_pad, axis=1),
                        "p_x_to_y_sem_by_pair": nan_sem(pxy_pad, axis=1),
                        "p_y_to_x_mean_by_pair": np.nanmean(pyx_pad, axis=1),
                        "p_y_to_x_sem_by_pair": nan_sem(pyx_pad, axis=1),
                        "p_x_to_y_mean": np.nanmean(pxy_pad[0], axis=0),
                        "p_x_to_y_sem": nan_sem(pxy_pad[0], axis=0),
                        "p_y_to_x_mean": np.nanmean(pyx_pad[0], axis=0),
                        "p_y_to_x_sem": nan_sem(pyx_pad[0], axis=0),
                    }
                )
                metrics.update(
                    {
                        "granger_mode": self.granger_mode,
                        "granger_n_pairs": float(n_pairs),
                        "granger_n_lags_tested": float(n_lag),
                        "granger_effective_max_lag_s": float(
                            granger_lag_ref[-1]
                        )
                        if granger_lag_ref.size
                        else np.nan,
                        "min_mean_p_x_to_y": float(
                            np.nanmin(arrays["p_x_to_y_mean"])
                        ),
                        "min_mean_p_y_to_x": float(
                            np.nanmin(arrays["p_y_to_x_mean"])
                        ),
                    }
                )

        return AnalysisResult(
            self.name,
            self.event_signal,
            self.window,
            self._params(channels, tuple(valid_pairs)),
            metrics,
            arrays,
            "peak-aligned multi-channel epochs with cross-correlation and optional Granger tests",
        )

    def _params(
        self,
        channels: Sequence[str] | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "event_signal": self.event_signal,
            "channels": list(channels)
            if channels is not None
            else self.channels,
            "connectivity_pairs": [list(p) for p in pairs]
            if pairs is not None
            else self.connectivity_pairs,
            "x_signal": self.x_signal,
            "y_signal": self.y_signal,
            "detector": self.detector,
            "detector_params": dict(self.detector_params or {}),
            "window": None if self.window is None else self.window.as_dict(),
            "t_before_s": self.t_before_s,
            "t_after_s": self.t_after_s,
            "target_fs": self.target_fs,
            "dt": self.dt,
            "baseline_window_s": self.baseline_window_s,
            "max_lag_s": self.max_lag_s,
            "detrend": self.detrend,
            "normalise": self.normalise,
            "run_granger": self.run_granger,
            "granger_mode": self.granger_mode,
            "granger_target_dt": self.granger_target_dt,
            "granger_max_lag_steps": self.granger_max_lag_steps,
            "granger_max_samples": self.granger_max_samples,
            "granger_max_events": self.granger_max_events,
        }

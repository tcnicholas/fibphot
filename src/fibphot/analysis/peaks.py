from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from ..stages.smooth import (
    KalmanModel,
    PadMode,
    WindowType,
    kalman_smooth_1d,
    savgol_smooth_1d,
    smooth_1d,
)
from ..state import PhotometryState
from ..types import FloatArray
from .report import AnalysisResult, AnalysisWindow

PeakKind = Literal["peak", "valley"]
SelectKind = Literal["peak", "valley", "both"]
AreaRegion = Literal["bases", "fwhm"]
BaselineMode = Literal["line", "flat"]

FitModelName = Literal["gaussian", "lorentzian", "alpha"]
SmoothMethod = Literal["moving", "savgol", "kalman"]
EdgeMethod = Literal["prominence", "fraction", "sigma"]


def _as_float_1d(x: FloatArray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return x


def _mad_sigma(x: FloatArray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _window_to_slice(state: PhotometryState, window: AnalysisWindow | None) -> slice:
    if window is None:
        return slice(0, state.n_samples)

    if window.ref == "samples":
        a = int(window.start)
        b = int(window.end)
        if a < 0 or b < 0:
            raise ValueError("Sample windows must be non-negative.")
        if b <= a:
            raise ValueError("Sample window must satisfy end > start.")
        return slice(max(0, a), min(state.n_samples, b))

    # seconds
    t0 = float(window.start)
    t1 = float(window.end)
    if t1 <= t0:
        raise ValueError("Seconds window must satisfy end > start.")
    t = state.time_seconds
    m = np.isfinite(t) & (t >= t0) & (t <= t1)
    if not np.any(m):
        return slice(0, 0)
    idx = np.where(m)[0]
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _fs_from_time(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    m = np.isfinite(t)
    if np.sum(m) < 3:
        return float("nan")
    dt = np.diff(t[m])
    if dt.size == 0:
        return float("nan")
    med = float(np.median(dt))
    if med <= 0:
        return float("nan")
    return 1.0 / med


def _interp_x_at_positions(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    idx = np.arange(x.shape[0], dtype=float)
    return np.interp(positions, idx, x)


def _baseline_flat(y: np.ndarray, left_i: int, right_i: int) -> float:
    return float(0.5 * (y[left_i] + y[right_i]))


def _baseline_line(x: np.ndarray, y: np.ndarray, left_i: int, right_i: int) -> np.ndarray:
    x0 = float(x[left_i])
    x1 = float(x[right_i])
    if np.isclose(x1, x0):
        return np.full_like(x, fill_value=float(y[left_i]), dtype=float)
    m = (float(y[right_i]) - float(y[left_i])) / (x1 - x0)
    c = float(y[left_i]) - m * x0
    return m * x + c


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))  # type: ignore[attr-defined]
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot < 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float, offset: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    return offset + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lorentzian(x: np.ndarray, amp: float, x0: float, gamma: float, offset: float) -> np.ndarray:
    gamma = max(float(gamma), 1e-12)
    return offset + amp * (gamma**2) / ((x - x0) ** 2 + gamma**2)


def alpha_transient(x: np.ndarray, amp: float, t0: float, tau: float, offset: float) -> np.ndarray:
    tau = max(float(tau), 1e-12)
    dt = (x - t0) / tau
    out = np.full_like(x, fill_value=offset, dtype=float)
    m = x >= t0
    out[m] = offset + amp * dt[m] * np.exp(1.0 - dt[m])
    return out


_MODEL_FUNCS: dict[FitModelName, Any] = {
    "gaussian": gaussian,
    "lorentzian": lorentzian,
    "alpha": alpha_transient,
}


@dataclass(frozen=True, slots=True)
class PeakFit:
    model: FitModelName
    params: tuple[float, ...]
    r2: float
    rmse: float
    success: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class PeakEvent:
    kind: PeakKind
    index: int
    x: float
    y: float

    prominence: float | None = None
    left_base_index: int | None = None
    right_base_index: int | None = None

    height: float | None = None
    fwhm: float | None = None
    left_ip: float | None = None
    right_ip: float | None = None

    # asymmetry helpers (derived from half-height crossings)
    rise_s: float | None = None
    decay_s: float | None = None

    area: float | None = None
    fit: PeakFit | None = None


def _fit_model_to_peak(
    x: np.ndarray,
    y: np.ndarray,
    peak: PeakEvent,
    *,
    model: FitModelName,
    window_s: float | None,
    window_samples: int | None,
    maxfev: int,
) -> PeakFit:
    import scipy.optimize

    n = x.shape[0]
    i0 = peak.index

    if window_samples is not None:
        half = max(int(window_samples // 2), 1)
        lo = max(0, i0 - half)
        hi = min(n, i0 + half + 1)
    elif window_s is not None:
        half = float(window_s) / 2.0
        lo_x = peak.x - half
        hi_x = peak.x + half
        lo = int(np.searchsorted(x, lo_x, side="left"))
        hi = int(np.searchsorted(x, hi_x, side="right"))
        lo, hi = max(0, lo), min(n, hi)
    else:
        lo = max(0, i0 - 50)
        hi = min(n, i0 + 51)

    xf = x[lo:hi]
    yf = y[lo:hi]
    if xf.size < 6:
        return PeakFit(
            model=model,
            params=(),
            r2=float("nan"),
            rmse=float("nan"),
            success=False,
            message="Not enough points.",
        )

    func = _MODEL_FUNCS[model]

    offset0 = float(np.median(yf))
    amp0 = float(peak.y - offset0)
    mu0 = float(peak.x)

    dx = float(np.median(np.diff(xf))) if xf.size > 2 else 1.0
    sigma0 = 3.0 * dx
    gamma0 = 3.0 * dx
    tau0 = 3.0 * dx

    try:
        if model == "gaussian":
            p0 = (amp0, mu0, sigma0, offset0)
            bounds = (
                (-np.inf, float(xf.min()), 1e-12, -np.inf),
                (np.inf, float(xf.max()), np.inf, np.inf),
            )
        elif model == "lorentzian":
            p0 = (amp0, mu0, gamma0, offset0)
            bounds = (
                (-np.inf, float(xf.min()), 1e-12, -np.inf),
                (np.inf, float(xf.max()), np.inf, np.inf),
            )
        else:  # alpha
            p0 = (amp0, mu0, tau0, offset0)
            bounds = (
                (-np.inf, float(xf.min()), 1e-12, -np.inf),
                (np.inf, float(xf.max()), np.inf, np.inf),
            )

        popt, _pcov = scipy.optimize.curve_fit(
            f=func,
            xdata=xf,
            ydata=yf,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )
        yhat = func(xf, *popt)
        return PeakFit(
            model=model,
            params=tuple(float(v) for v in popt),
            r2=_r2(yf, yhat),
            rmse=_rmse(yf, yhat),
            success=True,
        )
    except Exception as exc:  # noqa: BLE001
        return PeakFit(
            model=model,
            params=(),
            r2=float("nan"),
            rmse=float("nan"),
            success=False,
            message=str(exc),
        )


def _half_height_times(
    x: np.ndarray,
    y_corr: np.ndarray,
    peak_i: int,
    *,
    sign: float,
) -> tuple[float | None, float | None]:
    """
    Estimate rise/decay durations from half-height crossings.

    y_corr: baseline-corrected local segment (baseline ~ 0).
    sign: +1 for peaks, -1 for valleys (operate in "upward" space).
    """
    z = sign * y_corr
    if not np.isfinite(z[peak_i]):
        return None, None
    h = float(z[peak_i])
    if h <= 0:
        return None, None
    half = 0.5 * h

    left = None
    for i in range(peak_i, 0, -1):
        if np.isfinite(z[i]) and z[i] <= half:
            left = i
            break

    right = None
    for i in range(peak_i, z.size):
        if np.isfinite(z[i]) and z[i] <= half:
            right = i
            break

    if left is None or right is None:
        return None, None

    rise = float(x[peak_i] - x[left])
    decay = float(x[right] - x[peak_i])
    return rise, decay


def _edge_indices_from_threshold(
    z: np.ndarray,
    peak_i: int,
    thr: float,
) -> tuple[int | None, int | None]:
    """
    Find left/right indices where z drops to <= thr.
    z should be in "upward" space (peaks positive).
    """
    if not np.isfinite(thr):
        return None, None

    # guard: if thr is above peak height, it collapses to the peak itself
    h = float(z[peak_i]) if np.isfinite(z[peak_i]) else float("nan")
    if not np.isfinite(h) or thr >= h:
        return None, None

    li = None
    for i in range(peak_i, -1, -1):
        if np.isfinite(z[i]) and z[i] <= thr:
            li = i
            break

    ri = None
    for i in range(peak_i, z.size):
        if np.isfinite(z[i]) and z[i] <= thr:
            ri = i
            break

    return li, ri


@dataclass(frozen=True, slots=True)
class PeakAnalysis:
    """
    Peak/valley detection + measurements, packaged as an analysis object.

    - Detection can be done on a smoothed copy (recommended).
    - Measurements (area/height) are taken on the original unsmoothed trace.
    - Returns an AnalysisResult suitable for PhotometryReport.
    """

    signal: str
    kind: SelectKind = "peak"
    window: AnalysisWindow | None = None

    # Robustify detection:
    smooth_for_detection: bool = True
    smooth_method: SmoothMethod = "moving"

    # Moving-window smoothing options
    smooth_window_len: int = 25
    smooth_window: WindowType = "flat"
    smooth_pad_mode: PadMode = "reflect"
    smooth_match_edges: bool = True

    # SavGol options
    savgol_polyorder: int = 3
    savgol_mode: Literal["interp", "mirror", "nearest", "constant", "wrap"] = "interp"

    # Kalman options
    kalman_model: KalmanModel = "local_level"
    kalman_r: float | Literal["auto"] = "auto"
    kalman_q: float | None = None
    kalman_q_scale: float = 1e-3

    # scipy.signal.find_peaks parameters (in *signal units* / *samples*)
    height: float | tuple[float, float] | None = None
    prominence: float | tuple[float, float] | None = None
    threshold: float | tuple[float, float] | None = None
    wlen: int | None = None
    plateau_size: int | tuple[int, int] | None = None

    # convenience parameters in seconds (converted using fs)
    distance_s: float | None = 0.25
    width_s: float | None = 0.8

    # if explicit sample-based distance/width are set, they override *_s
    distance: int | None = None
    width: float | tuple[float, float] | None = None

    # auto thresholds based on MAD(y_det) if height/prominence are None
    auto_height_sigmas: float = 1.0
    auto_prominence_sigmas: float = 2.0

    # measurement choices
    rel_height: float = 0.5
    baseline_mode: BaselineMode = "line"
    area_region: AreaRegion = "bases"

    # edge selection (helps long decay peaks)
    edge_method: EdgeMethod = "prominence"
    edge_fraction: float = 0.10      # used when edge_method="fraction"
    edge_sigmas: float = 1.0         # used when edge_method="sigma"

    # optional per-peak parametric fit (helpful for asymmetric peaks)
    fit_model: FitModelName | None = "alpha"
    fit_window_s: float | None = 2.0
    fit_window_samples: int | None = None
    fit_maxfev: int = 5000

    def __call__(self, state: PhotometryState) -> AnalysisResult:
        import scipy.signal

        i_sig = state.idx(self.signal)
        t_full = _as_float_1d(state.time_seconds)
        y_raw_full = _as_float_1d(state.signals[i_sig])

        sl = _window_to_slice(state, self.window)
        t = t_full[sl]
        y_raw = y_raw_full[sl]

        if t.size < 4:
            return AnalysisResult(
                name="peaks",
                channel=self.signal,
                window=self.window,
                params=self._params(),
                metrics={"n_events": 0.0},
                arrays={},
                notes="Empty/too-small window.",
            )

        fs = _fs_from_time(t)
        if not np.isfinite(fs) or fs <= 0:
            fs = float(state.sampling_rate)

        # detection signal (optionally smoothed)
        y_det = y_raw.copy()
        if self.smooth_for_detection:
            if self.smooth_method == "moving":
                y_det = smooth_1d(
                    y_det,
                    window_len=self.smooth_window_len,
                    window=self.smooth_window,
                    pad_mode=self.smooth_pad_mode,
                    match_edges=self.smooth_match_edges,
                )
            elif self.smooth_method == "savgol":
                y_det = savgol_smooth_1d(
                    y_det,
                    window_len=self.smooth_window_len,
                    polyorder=self.savgol_polyorder,
                    mode=self.savgol_mode,
                )
            elif self.smooth_method == "kalman":
                y_det = kalman_smooth_1d(
                    y_det,
                    model=self.kalman_model,
                    r=self.kalman_r,
                    q=self.kalman_q,
                    q_scale=self.kalman_q_scale,
                )
            else:
                raise ValueError(f"Unknown smooth_method: {self.smooth_method!r}")

        # robust sigma estimate for auto thresholds and optional edge_method="sigma"
        sigma = _mad_sigma(y_det - np.nanmedian(y_det))

        # auto thresholds on detection signal
        height = self.height
        prominence = self.prominence
        if height is None and np.isfinite(sigma):
            height = float(self.auto_height_sigmas) * float(sigma)
        if prominence is None and np.isfinite(sigma):
            prominence = float(self.auto_prominence_sigmas) * float(sigma)

        # convert seconds -> samples for find_peaks
        distance = self.distance
        width = self.width

        if distance is None and self.distance_s is not None:
            distance = int(max(1, round(float(self.distance_s) * fs)))

        if width is None and self.width_s is not None:
            width = float(max(1.0, round(float(self.width_s) * fs)))

        base_kwargs: dict[str, Any] = {
            "height": height,
            "prominence": prominence,
            "threshold": self.threshold,
            "distance": distance,
            "width": width,
            "wlen": self.wlen,
            "plateau_size": self.plateau_size,
        }
        base_kwargs = {k: v for k, v in base_kwargs.items() if v is not None}

        events: list[PeakEvent] = []

        def detect_one(kind: PeakKind) -> None:
            if kind == "peak":
                y_work = y_det
                sign = 1.0
            else:
                y_work = -y_det
                sign = -1.0

            finite = np.isfinite(y_work)
            if np.sum(finite) < 5:
                return

            # interpolate NaNs for scipy
            yw = y_work.copy()
            if not np.all(finite):
                xi = np.arange(yw.size, dtype=float)
                yw[~finite] = np.interp(xi[~finite], xi[finite], yw[finite])

            idx, _props = scipy.signal.find_peaks(yw, **base_kwargs)
            if idx.size == 0:
                return

            # prominence bases (in detection space)
            prom, left_bases0, right_bases0 = scipy.signal.peak_prominences(
                yw, idx, wlen=self.wlen
            )

            # widths at rel_height (FWHM if rel_height=0.5)
            widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(
                yw, idx, rel_height=self.rel_height, wlen=self.wlen
            )

            # convert fractional sample positions to x-units
            left_ip_x = _interp_x_at_positions(t, left_ips.astype(float))
            right_ip_x = _interp_x_at_positions(t, right_ips.astype(float))

            for j, i0 in enumerate(idx.tolist()):
                lb0 = int(left_bases0[j])
                rb0 = int(right_bases0[j])

                # First-pass baseline from prominence bases
                if self.baseline_mode == "flat":
                    b0 = _baseline_flat(y_raw, lb0, rb0)
                    base_line0 = None
                    ycorr0 = y_raw - b0
                else:
                    base_line0 = _baseline_line(t, y_raw, lb0, rb0)
                    b0 = float(base_line0[i0])
                    ycorr0 = y_raw - base_line0

                height0 = float(y_raw[i0] - b0)

                # Choose final "bases" (event edges)
                lb = lb0
                rb = rb0

                if self.edge_method != "prominence":
                    # operate in upward space
                    z = sign * ycorr0
                    zheight = float(sign * height0)

                    if self.edge_method == "fraction":
                        frac = float(self.edge_fraction)
                        if not (0.0 < frac < 1.0):
                            raise ValueError("edge_fraction must be in (0, 1).")
                        thr = frac * zheight
                    else:  # sigma
                        thr = float(self.edge_sigmas) * float(sigma) if np.isfinite(sigma) else float("nan")

                    li, ri = _edge_indices_from_threshold(z, i0, thr)
                    if li is not None:
                        lb = int(li)
                    if ri is not None:
                        rb = int(ri)

                # Second-pass baseline using chosen lb/rb
                b: float | None = None
                if self.baseline_mode == "flat":
                    b = _baseline_flat(y_raw, lb, rb)
                    base_line = None
                    base_at_peak = float(b)
                else:
                    base_line = _baseline_line(t, y_raw, lb, rb)
                    base_at_peak = float(base_line[i0])

                height_raw = float(y_raw[i0] - base_at_peak)

                # region for area
                if self.area_region == "fwhm":
                    lo_x = float(left_ip_x[j])
                    hi_x = float(right_ip_x[j])
                    lo = int(np.searchsorted(t, lo_x, side="left"))
                    hi = int(np.searchsorted(t, hi_x, side="right"))
                    lo, hi = max(0, lo), min(t.size, hi)
                else:
                    lo = min(lb, rb)
                    hi = max(lb, rb) + 1

                # baseline-corrected area
                if hi - lo >= 2:
                    xs = t[lo:hi]
                    ys = y_raw[lo:hi]
                    if self.baseline_mode == "flat":
                        assert b is not None
                        ycorr = ys - b
                    else:
                        assert base_line is not None
                        ycorr = ys - base_line[lo:hi]
                    area = _trapz(ycorr, xs)
                else:
                    area = float("nan")

                # FWHM in x-units (from detection)
                fwhm = float(right_ip_x[j] - left_ip_x[j])

                # asymmetry: compute on local baseline-corrected segment (use lb/rb)
                lo2 = min(lb, rb)
                hi2 = max(lb, rb) + 1
                rise_s = decay_s = None
                if hi2 - lo2 >= 5:
                    xs2 = t[lo2:hi2]
                    ys2 = y_raw[lo2:hi2]
                    if self.baseline_mode == "flat":
                        assert b is not None
                        ycorr2 = ys2 - b
                    else:
                        assert base_line is not None
                        ycorr2 = ys2 - base_line[lo2:hi2]
                    peak_i_local = int(i0 - lo2)
                    r, d = _half_height_times(xs2, ycorr2, peak_i_local, sign=1.0 if kind == "peak" else -1.0)
                    rise_s, decay_s = r, d

                ev = PeakEvent(
                    kind=kind,
                    index=int(i0),
                    x=float(t[i0]),
                    y=float(y_raw[i0]),
                    prominence=float(prom[j]),
                    left_base_index=int(lb),
                    right_base_index=int(rb),
                    height=float(height_raw),
                    fwhm=float(fwhm),
                    left_ip=float(left_ip_x[j]),
                    right_ip=float(right_ip_x[j]),
                    rise_s=rise_s,
                    decay_s=decay_s,
                    area=float(area),
                    fit=None,
                )

                if self.fit_model is not None:
                    fit = _fit_model_to_peak(
                        t,
                        y_raw,
                        ev,
                        model=self.fit_model,
                        window_s=self.fit_window_s,
                        window_samples=self.fit_window_samples,
                        maxfev=self.fit_maxfev,
                    )
                    ev = replace(ev, fit=fit)

                events.append(ev)

        if self.kind in ("peak", "both"):
            detect_one("peak")
        if self.kind in ("valley", "both"):
            detect_one("valley")

        events.sort(key=lambda e: e.index)

        arrays = _events_to_arrays(events, offset=int(sl.start))
        metrics = _events_to_metrics(events)

        notes = (
            f"detection: {'smoothed' if self.smooth_for_detection else 'raw'} "
            f"({self.smooth_method}) ; measurements: raw ; "
            f"edges: {self.edge_method}"
        )

        return AnalysisResult(
            name="peaks",
            channel=self.signal,
            window=self.window,
            params=self._params(),
            metrics=metrics,
            arrays=arrays,
            notes=notes,
        )

    def _params(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "kind": self.kind,
            "window": None if self.window is None else {
                "start": self.window.start,
                "end": self.window.end,
                "ref": self.window.ref,
                "label": self.window.label,
            },
            "smooth_for_detection": self.smooth_for_detection,
            "smooth_method": self.smooth_method,
            "smooth_window_len": self.smooth_window_len,
            "smooth_window": self.smooth_window,
            "smooth_pad_mode": self.smooth_pad_mode,
            "smooth_match_edges": self.smooth_match_edges,
            "savgol_polyorder": self.savgol_polyorder,
            "savgol_mode": self.savgol_mode,
            "kalman_model": self.kalman_model,
            "kalman_r": self.kalman_r,
            "kalman_q": self.kalman_q,
            "kalman_q_scale": self.kalman_q_scale,
            "distance_s": self.distance_s,
            "width_s": self.width_s,
            "rel_height": self.rel_height,
            "baseline_mode": self.baseline_mode,
            "area_region": self.area_region,
            "auto_height_sigmas": self.auto_height_sigmas,
            "auto_prominence_sigmas": self.auto_prominence_sigmas,
            "edge_method": self.edge_method,
            "edge_fraction": self.edge_fraction,
            "edge_sigmas": self.edge_sigmas,
            "fit_model": self.fit_model,
            "fit_window_s": self.fit_window_s,
            "fit_window_samples": self.fit_window_samples,
        }


def _events_to_arrays(events: Sequence[PeakEvent], offset: int = 0) -> dict[str, np.ndarray]:
    if len(events) == 0:
        return {}

    kind = np.array([e.kind for e in events], dtype="U6")
    index = np.array([e.index + offset for e in events], dtype=int)
    x = np.array([e.x for e in events], dtype=float)
    y = np.array([e.y for e in events], dtype=float)

    prominence = np.array(
        [np.nan if e.prominence is None else float(e.prominence) for e in events],
        dtype=float,
    )
    left_base = np.array(
        [(-1 if e.left_base_index is None else int(e.left_base_index) + offset) for e in events],
        dtype=int,
    )
    right_base = np.array(
        [(-1 if e.right_base_index is None else int(e.right_base_index) + offset) for e in events],
        dtype=int,
    )

    height = np.array([np.nan if e.height is None else float(e.height) for e in events], dtype=float)
    fwhm = np.array([np.nan if e.fwhm is None else float(e.fwhm) for e in events], dtype=float)
    left_ip = np.array([np.nan if e.left_ip is None else float(e.left_ip) for e in events], dtype=float)
    right_ip = np.array([np.nan if e.right_ip is None else float(e.right_ip) for e in events], dtype=float)
    rise_s = np.array([np.nan if e.rise_s is None else float(e.rise_s) for e in events], dtype=float)
    decay_s = np.array([np.nan if e.decay_s is None else float(e.decay_s) for e in events], dtype=float)
    area = np.array([np.nan if e.area is None else float(e.area) for e in events], dtype=float)

    fit_success = np.array([False if e.fit is None else bool(e.fit.success) for e in events], dtype=bool)
    fit_r2 = np.array([np.nan if e.fit is None else float(e.fit.r2) for e in events], dtype=float)
    fit_rmse = np.array([np.nan if e.fit is None else float(e.fit.rmse) for e in events], dtype=float)
    fit_model = np.array(["" if e.fit is None else str(e.fit.model) for e in events], dtype="U12")

    # force 1D object array for params (avoids pandas “ndim > 1” error)
    fit_params = np.empty(len(events), dtype=object)
    for i, e in enumerate(events):
        fit_params[i] = () if e.fit is None else tuple(e.fit.params)

    return {
        "kind": kind,
        "index": index,
        "x": x,
        "y": y,
        "prominence": prominence,
        "left_base_index": left_base,
        "right_base_index": right_base,
        "height": height,
        "fwhm": fwhm,
        "left_ip": left_ip,
        "right_ip": right_ip,
        "rise_s": rise_s,
        "decay_s": decay_s,
        "area": area,
        "fit_success": fit_success,
        "fit_r2": fit_r2,
        "fit_rmse": fit_rmse,
        "fit_model": fit_model,
        "fit_params": fit_params,
    }


def _events_to_metrics(events: Sequence[PeakEvent]) -> dict[str, float]:
    if len(events) == 0:
        return {"n_events": 0.0}

    heights = np.array([np.nan if e.height is None else float(e.height) for e in events], dtype=float)
    areas = np.array([np.nan if e.area is None else float(e.area) for e in events], dtype=float)
    fwhm = np.array([np.nan if e.fwhm is None else float(e.fwhm) for e in events], dtype=float)

    return {
        "n_events": float(len(events)),
        "mean_height": float(np.nanmean(heights)) if np.any(np.isfinite(heights)) else float("nan"),
        "mean_area": float(np.nanmean(areas)) if np.any(np.isfinite(areas)) else float("nan"),
        "mean_fwhm": float(np.nanmean(fwhm)) if np.any(np.isfinite(fwhm)) else float("nan"),
    }


def peak_result_to_dataframe(res: AnalysisResult):
    """
    Convert AnalysisResult.arrays to a pandas DataFrame.

    Robust to:
    - object columns (e.g. fit_params tuples)
    - accidental multi-dimensional arrays (converted row-wise to tuples/objects)
    """
    import pandas as pd

    a = res.arrays
    if not a:
        return pd.DataFrame()

    cols: dict[str, Any] = {}
    n: int | None = None

    for k, v in a.items():
        vv = v
        if isinstance(vv, np.ndarray):
            if vv.ndim == 0:
                vv = np.array([vv.item()])
            elif vv.ndim == 1:
                pass
            elif vv.ndim == 2:
                # convert each row to a tuple -> 1D object column
                vv = [tuple(row.tolist()) for row in vv]
            else:
                # higher dims: store each entry as an object
                vv = [vv[i] for i in range(vv.shape[0])]

        if n is None:
            n = len(vv)
        else:
            if len(vv) != n:
                raise ValueError(
                    f"Column {k!r} has length {len(vv)} but expected {n}."
                )

        cols[k] = vv

    return pd.DataFrame(cols)


def plot_peak_result(
    state: PhotometryState,
    res: AnalysisResult,
    *,
    label: str | None = None,
    show_window: bool = True,
    show_bases: bool = False,
    show_fwhm: bool = True,
    annotate: bool = False,
    show_area: bool = False,
    area_event: int | None = None,             # which event index to highlight
    area_region_override: AreaRegion | None = None,  # "bases" or "fwhm"
    area_alpha: float = 0.25,
    ax=None,
):
    import matplotlib.pyplot as plt

    sig = res.channel
    t = np.asarray(state.time_seconds, dtype=float)
    y = np.asarray(state.channel(sig), dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    else:
        fig = ax.figure

    ax.plot(t, y, linewidth=1.2, alpha=0.9, label=label or sig)

    if show_window and res.window is not None and res.window.ref == "seconds":
        ax.axvspan(float(res.window.start), float(res.window.end), alpha=0.06, color="gray")

    a = res.arrays
    if not a:
        ax.legend(frameon=False, fontsize=8)
        return fig, ax

    kinds = a.get("kind")
    xs = a.get("x")
    ys = a.get("y")
    if kinds is None or xs is None or ys is None:
        ax.legend(frameon=False, fontsize=8)
        return fig, ax

    # markers
    m_peak = (kinds == "peak")
    m_valley = (kinds == "valley")

    if np.any(m_peak):
        ax.scatter(xs[m_peak], ys[m_peak], s=18, linewidth=0.0, label="peaks")
    if np.any(m_valley):
        ax.scatter(xs[m_valley], ys[m_valley], s=18, linewidth=0.0, label="valleys")

    # bases markers
    if show_bases:
        lb = a.get("left_base_index")
        rb = a.get("right_base_index")
        if lb is not None and rb is not None:
            for i in range(len(xs)):
                lbi = int(lb[i])
                rbi = int(rb[i])
                if lbi >= 0 and rbi >= 0:
                    ax.scatter([t[lbi], t[rbi]], [y[lbi], y[rbi]], marker="x", s=30)

    # FWHM guides
    if show_fwhm:
        li = a.get("left_ip")
        ri = a.get("right_ip")
        if li is not None and ri is not None:
            for i in range(len(xs)):
                if np.isfinite(li[i]) and np.isfinite(ri[i]):
                    ax.vlines(
                        [float(li[i]), float(ri[i])],
                        ymin=np.nanmin(y),
                        ymax=np.nanmax(y),
                        linestyles="dashed",
                        alpha=0.25,
                    )

    # draw the integration region used for "area" for one chosen event
    if show_area:
        lb = a.get("left_base_index")
        rb = a.get("right_base_index")
        li = a.get("left_ip")
        ri = a.get("right_ip")
        area = a.get("area")

        if lb is not None and rb is not None and len(xs) > 0:
            # choose which event to highlight
            if area_event is None:
                # default: largest |area| if available, else first event
                if area is not None and np.any(np.isfinite(area)):
                    area_event = int(np.nanargmax(np.abs(area)))
                else:
                    area_event = 0

            i = int(area_event)
            if 0 <= i < len(xs):
                lbi = int(lb[i]) if lb is not None else -1
                rbi = int(rb[i]) if rb is not None else -1

                if lbi >= 0 and rbi >= 0:
                    # which region do we shade?
                    params = res.params or {}
                    area_region = area_region_override or params.get("area_region", "bases")
                    baseline_mode = params.get("baseline_mode", "line")

                    if area_region == "fwhm" and li is not None and ri is not None and np.isfinite(li[i]) and np.isfinite(ri[i]):
                        lo = int(np.searchsorted(t, float(li[i]), side="left"))
                        hi = int(np.searchsorted(t, float(ri[i]), side="right"))
                    else:
                        lo = min(lbi, rbi)
                        hi = max(lbi, rbi) + 1

                    lo = max(0, lo)
                    hi = min(t.size, hi)

                    if hi - lo >= 2:
                        xs_area = t[lo:hi]
                        ys_area = y[lo:hi]

                        # baseline evaluated over the same xs_area
                        if baseline_mode == "flat":
                            b0 = _baseline_flat(y, lbi, rbi)
                            b = np.full_like(ys_area, b0, dtype=float)
                        else:
                            b_line = _baseline_line(t, y, lbi, rbi)
                            b = b_line[lo:hi]

                        m = np.isfinite(xs_area) & np.isfinite(ys_area) & np.isfinite(b)
                        if np.any(m):
                            # fill between baseline and trace over the integration region
                            ax.fill_between(
                                xs_area[m],
                                b[m],
                                ys_area[m],
                                alpha=area_alpha,
                                label="area region" if "area region" not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else None,
                            )

                            # show bounds + baseline segment (helps debugging “why did it stop early?”)
                            ax.vlines([xs_area[m][0], xs_area[m][-1]],
                                      ymin=np.nanmin(y), ymax=np.nanmax(y),
                                      linestyles=":", alpha=0.35)
                            ax.plot(xs_area[m], b[m], linewidth=1.0, alpha=0.6)

                            # annotate area value (if present)
                            if area is not None and np.isfinite(area[i]):
                                ax.annotate(
                                    f"area={float(area[i]):.3g}",
                                    (float(xs[i]), float(ys[i])),
                                    textcoords="offset points",
                                    xytext=(6, -10),
                                    fontsize=8,
                                )

    if annotate:
        for i in range(len(xs)):
            ax.annotate(
                f"{kinds[i]}@{xs[i]:.2f}",
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_xlabel("time (s)")
    ax.set_ylabel(sig)
    ax.legend(frameon=False, fontsize=8)
    return fig, ax

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..state import PhotometryState
from ..types import FloatArray
from .report import AnalysisResult, AnalysisWindow

AUCMode = Literal["signed", "positive", "negative", "absolute"]
BaselineRef = Literal[
    "zero",
    "window_mean",
    "window_median",
    "pre_mean",
    "pre_median",
    "window_quantile",
    "pre_quantile",
]


def _trapz(y: FloatArray, x: FloatArray) -> float:
    # numpy >= 2.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))  # type: ignore[attr-defined]
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


def _as_float_array(x: FloatArray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _resolve_window_seconds(
    state: PhotometryState,
    window: AnalysisWindow | None,
) -> tuple[np.ndarray, float, float, int, int]:
    """
    Returns:
      mask: bool array over samples (finite points inside window)
      t0, t1: window bounds in seconds (inclusive endpoints for display)
      i0, i1: window bounds in samples [i0, i1) for pre-window logic
    """
    t = _as_float_array(state.time_seconds)
    n = int(t.size)

    if n == 0:
        raise ValueError("State has zero samples.")

    if window is None:
        # Whole trace (finite points only)
        mask = np.isfinite(t)
        i0, i1 = 0, n
        t0, t1 = float(t[0]), float(t[-1])
        return mask, t0, t1, i0, i1

    if window.ref == "seconds":
        t0 = float(window.start)
        t1 = float(window.end)
        if t0 > t1:
            raise ValueError(f"Expected start <= end, got {t0} > {t1}.")

        mask = np.isfinite(t) & (t >= t0) & (t <= t1)

        # i0/i1 are best-effort sample bounds for the pre-window (not exact if irregular t)
        # We pick the first/last included sample indices.
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            # window empty
            return mask, t0, t1, 0, 0
        i0 = int(idx[0])
        i1 = int(idx[-1] + 1)
        return mask, t0, t1, i0, i1

    if window.ref == "samples":
        i0 = int(window.start)
        i1 = int(window.end)
        if i0 < 0 or i1 < 0:
            raise ValueError("Sample windows must be non-negative.")
        if i0 >= i1:
            raise ValueError(f"Expected start < end for samples, got {i0} >= {i1}.")
        i0 = max(0, min(n, i0))
        i1 = max(0, min(n, i1))

        mask = np.isfinite(t)
        mask &= np.zeros(n, dtype=bool)
        mask[i0:i1] = True

        # for display, map to time bounds if possible
        t0 = float(t[i0]) if i0 < n else float(t[-1])
        t1 = float(t[i1 - 1]) if (i1 - 1) < n and i1 > 0 else float(t[-1])
        return mask, t0, t1, i0, i1

    raise ValueError(f"Unknown window.ref: {window.ref!r}")


def _baseline_value(
    state: PhotometryState,
    y: FloatArray,
    *,
    window_mask: np.ndarray,
    t0: float,
    t1: float,
    i0: int,
    i1: int,
    baseline: BaselineRef,
    pre_seconds: float,
    quantile: float,
) -> float:
    """
    baseline options:
      - window_* computed over the analysis window
      - pre_* computed over [t0-pre_seconds, t0] OR [i0-pre_samples, i0)
        and falls back to window if empty
    """
    t = _as_float_array(state.time_seconds)
    y = _as_float_array(y)

    if baseline == "zero":
        return 0.0

    def _stat(vals: np.ndarray) -> float:
        if vals.size == 0:
            return float("nan")
        if baseline.endswith("mean"):
            return float(np.nanmean(vals))
        if baseline.endswith("median"):
            return float(np.nanmedian(vals))
        if baseline.endswith("quantile"):
            q = float(quantile)
            if not (0.0 <= q <= 1.0):
                raise ValueError("quantile must be in [0, 1].")
            return float(np.nanquantile(vals, q))
        raise ValueError(f"Unhandled baseline option: {baseline!r}")

    if baseline.startswith("window_"):
        m = window_mask & np.isfinite(y)
        return _stat(y[m])

    if baseline.startswith("pre_"):
        # Prefer seconds logic (more robust to variable dt), but we also have sample bounds.
        pre0 = t0 - float(pre_seconds)
        pre1 = t0

        m = np.isfinite(t) & np.isfinite(y) & (t >= pre0) & (t <= pre1)

        # If pre-window is empty (event near start), fall back to sample-based pre,
        # then to window.
        if int(np.sum(m)) < 2:
            if state.n_samples > 1 and np.isfinite(state.sampling_rate) and state.sampling_rate > 0:
                pre_n = int(round(float(pre_seconds) * float(state.sampling_rate)))
            else:
                pre_n = 0

            if pre_n > 0 and i0 > 0:
                j0 = max(0, i0 - pre_n)
                j1 = i0
                m = np.isfinite(t) & np.isfinite(y)
                mm = np.zeros_like(m, dtype=bool)
                mm[j0:j1] = True
                m &= mm

        if int(np.sum(m)) < 2:
            m = window_mask & np.isfinite(y)

        return _stat(y[m])

    raise ValueError(f"Unknown baseline: {baseline!r}")


def _contrib(u: np.ndarray, mode: AUCMode) -> np.ndarray:
    if mode == "signed":
        return u
    if mode == "positive":
        return np.clip(u, 0.0, np.inf)
    if mode == "negative":
        return np.clip(u, -np.inf, 0.0)
    if mode == "absolute":
        return np.abs(u)
    raise ValueError(f"Unknown mode: {mode!r}")


def auc_window(
    t: FloatArray,
    y: FloatArray,
    *,
    t0: float,
    t1: float,
    mode: AUCMode = "positive",
    baseline_value: float = 0.0,
) -> dict[str, float]:
    """
    Pure numeric AUC over [t0, t1] relative to a *provided* baseline value.
    (Convenient for unit tests / standalone use.)
    """
    t = _as_float_array(t)
    y = _as_float_array(y)

    if t0 > t1:
        raise ValueError(f"Expected t0 <= t1, got {t0} > {t1}.")

    m = np.isfinite(t) & np.isfinite(y) & (t >= t0) & (t <= t1)
    tt = t[m]
    yy = y[m]
    if tt.size < 2:
        return {"auc": float("nan"), "mean": float("nan"), "n_points": float(tt.size)}

    u = yy - float(baseline_value)
    c = _contrib(u, mode)
    area = _trapz(c, tt)

    duration = float(tt[-1] - tt[0])
    mean_amp = area / duration if duration > 0 and np.isfinite(duration) else float("nan")

    return {"auc": float(area), "mean": float(mean_amp), "n_points": float(tt.size)}


@dataclass(frozen=True, slots=True)
class AUC:
    """
    Windowed AUC analysis that returns an AnalysisResult.

    Example
    -------
    res = AUC(
        signal="gcamp",
        window=AnalysisWindow(0, 400, ref="seconds", label="cue window"),
        mode="positive",
        baseline="pre_median",
        pre_seconds=10.0,
    )(state)

    Then attach to a report:
        report = PhotometryReport(state).add(res)
    """

    signal: str
    window: AnalysisWindow | None = None
    mode: AUCMode = "positive"
    baseline: BaselineRef = "pre_median"
    pre_seconds: float = 10.0
    quantile: float = 0.1

    name: str = "auc"

    def __call__(self, state: PhotometryState) -> AnalysisResult:
        t = _as_float_array(state.time_seconds)
        y = _as_float_array(state.channel(self.signal))

        window_mask, t0, t1, i0, i1 = _resolve_window_seconds(state, self.window)

        # windowed samples
        m = window_mask & np.isfinite(y) & np.isfinite(t)
        tt = t[m]
        yy = y[m]

        if tt.size < 2:
            return AnalysisResult(
                name=self.name,
                channel=self.signal.lower(),
                window=self.window,
                params={
                    "mode": self.mode,
                    "baseline": self.baseline,
                    "pre_seconds": float(self.pre_seconds),
                    "quantile": float(self.quantile),
                },
                metrics={
                    "auc": float("nan"),
                    "mean": float("nan"),
                    "baseline": float("nan"),
                    "t0": float(t0),
                    "t1": float(t1),
                    "duration_s": float(t1 - t0),
                    "n_points": float(tt.size),
                },
                arrays={},
                notes="Window contained <2 finite points; AUC not computed.",
            )

        b = _baseline_value(
            state,
            y,
            window_mask=window_mask,
            t0=t0,
            t1=t1,
            i0=i0,
            i1=i1,
            baseline=self.baseline,
            pre_seconds=self.pre_seconds,
            quantile=self.quantile,
        )

        u = yy - b
        contrib = _contrib(u, self.mode)

        area = _trapz(contrib, tt)
        duration = float(tt[-1] - tt[0])
        mean_amp = area / duration if duration > 0 and np.isfinite(duration) else float("nan")

        return AnalysisResult(
            name=self.name,
            channel=self.signal.lower(),
            window=self.window,
            params={
                "mode": self.mode,
                "baseline": self.baseline,
                "pre_seconds": float(self.pre_seconds),
                "quantile": float(self.quantile),
            },
            metrics={
                "auc": float(area),
                "mean": float(mean_amp),
                "baseline": float(b),
                "t0": float(t0),
                "t1": float(t1),
                "duration_s": float(t1 - t0),
                "n_points": float(tt.size),
            },
            arrays={
                # Small, window-only arrays: safe to store and perfect for plotting/QA
                "t_window": tt,
                "y_window": yy,
                "u_window": u,
                "contrib": contrib,
            },
        )
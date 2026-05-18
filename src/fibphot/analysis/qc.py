from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..state import PhotometryState
from .report import AnalysisResult


@dataclass(frozen=True, slots=True)
class QCAnalysis:
    signal: str | None = None
    saturation_low: float | None = None
    saturation_high: float | None = None

    def __call__(self, state: PhotometryState) -> AnalysisResult:
        t = np.asarray(state.time_seconds, dtype=float)
        if self.signal is None:
            y = np.asarray(state.signals, dtype=float).ravel()
            channel = "all"
        else:
            y = np.asarray(state.channel(self.signal), dtype=float)
            channel = self.signal.lower()
        finite_y = np.isfinite(y)
        dt = np.diff(t)
        dt_good = dt[np.isfinite(dt) & (dt > 0)]
        metrics: dict[str, float] = {
            "n_samples": float(state.n_samples),
            "n_signals": float(state.n_signals),
            "duration_s": float(t[-1] - t[0]) if t.size >= 2 else float("nan"),
            "sampling_rate_hz": float(state.sampling_rate),
            "time_nan_fraction": float(np.mean(~np.isfinite(t))) if t.size else float("nan"),
            "signal_nan_fraction": float(np.mean(~finite_y)) if y.size else float("nan"),
            "signal_mean": float(np.nanmean(y)) if np.any(finite_y) else float("nan"),
            "signal_std": float(np.nanstd(y)) if np.any(finite_y) else float("nan"),
            "signal_mad_sigma": _mad_sigma(y),
        }
        if dt_good.size:
            med = float(np.median(dt_good))
            metrics["dt_median_s"] = med
            metrics["dt_max_rel_deviation"] = float(np.nanmax(np.abs(dt_good - med) / med)) if med > 0 else float("nan")
        if self.saturation_low is not None:
            metrics["saturation_low_fraction"] = float(np.mean(y <= self.saturation_low))
        if self.saturation_high is not None:
            metrics["saturation_high_fraction"] = float(np.mean(y >= self.saturation_high))
        return AnalysisResult(name="qc", channel=channel, window=None, metrics=metrics, params=self._params())

    def _params(self) -> dict[str, Any]:
        return {"signal": self.signal, "saturation_low": self.saturation_low, "saturation_high": self.saturation_high}


def _mad_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))

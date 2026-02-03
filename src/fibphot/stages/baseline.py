from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize
import scipy.signal

from ..state import PhotometryState
from ..types import FloatArray
from .base import StageOutput, UpdateStage, _resolve_channels


def double_exponential(
    times: FloatArray,
    const: float,
    amp_fast: float,
    amp_slow: float,
    tau_fast: float,
    tau_slow: float,
) -> FloatArray:
    """ Double exponential function for baseline fitting. """
    t = np.asarray(times, dtype=float)
    return const + amp_slow*np.exp(-t/tau_slow) + amp_fast*np.exp(-t/tau_fast)


def _r2(y: FloatArray, yhat: FloatArray) -> float:
    """ Coefficient of determination R^2. """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y: FloatArray, yhat: FloatArray) -> float:
    """ Root mean square error. """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _initial_guess(y: FloatArray) -> list[float]:
    """ Initial guess for double exponential fitting. """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    tail = y[int(n * 0.9) :] if n >= 10 else y
    const = float(np.median(tail))
    amp = float(max(y[0] - const, 1e-6))
    return [const, amp * 0.6, amp * 0.4, 300.0, 3000.0]


@dataclass(frozen=True, slots=True)
class DoubleExpBaseline(UpdateStage):
    name: str = field(default="double_exp_baseline", init=False)

    subtract: bool = False
    channels: str | list[str] | None = None
    decimate_to_hz: float | None = None
    maxfev: int = 2000

    tau_fast_bounds: tuple[float, float] = (60.0, 600.0)
    tau_slow_bounds: tuple[float, float] = (600.0, 36000.0)

    def _params_for_summary(self) -> dict[str, object]:
        return {
            "subtract": self.subtract,
            "channels": self.channels if self.channels is not None else "all",
            "decimate_to_hz": self.decimate_to_hz,
            "maxfev": self.maxfev,
            "tau_fast_bounds": self.tau_fast_bounds,
            "tau_slow_bounds": self.tau_slow_bounds,
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        idxs = _resolve_channels(state, self.channels)
        t_full = state.time_seconds

        baseline = np.zeros_like(state.signals)
        params_out = np.full((state.n_signals, 5), np.nan, dtype=float)
        r2_out = np.full((state.n_signals,), np.nan, dtype=float)
        rmse_out = np.full((state.n_signals,), np.nan, dtype=float)

        fs = state.sampling_rate
        if self.decimate_to_hz is None or self.decimate_to_hz <= 0:
            decim = 1
        else:
            decim = max(1, int(round(fs / float(self.decimate_to_hz))))

        for i in idxs:
            y_full = state.signals[i]

            if decim > 1:

                # decimate both for fitting.
                y_fit = scipy.signal.decimate(
                    y_full, decim, ftype="fir", zero_phase=True
                )
                t_fit = t_full[::decim]

                # align lengths conservatively
                m = min(t_fit.shape[0], y_fit.shape[0])
                t_fit = t_fit[:m]
                y_fit = y_fit[:m]
            else:
                t_fit = t_full
                y_fit = y_full

            guess = _initial_guess(y_fit)

            y_max = float(np.max(y_fit))
            lo = [
                0.0, 0.0, 0.0,
                self.tau_fast_bounds[0], self.tau_slow_bounds[0]
            ]
            hi = [
                y_max, y_max, y_max, 
                self.tau_fast_bounds[1], self.tau_slow_bounds[1]
            ]

            popt, _ = scipy.optimize.curve_fit(
                f=double_exponential,
                xdata=t_fit,
                ydata=y_fit,
                p0=guess,
                bounds=(lo, hi),
                maxfev=self.maxfev,
            )

            yhat_fit = double_exponential(t_fit, *popt)
            r2_out[i] = _r2(y_fit, yhat_fit)
            rmse_out[i] = _rmse(y_fit, yhat_fit)

            params_out[i] = popt
            baseline[i] = double_exponential(t_full, *popt)

        new_signals = state.signals.copy()
        if self.subtract:
            for i in idxs:
                new_signals[i] = new_signals[i] - baseline[i]

        metrics = {
            "mean_r2": float(
                np.nanmean(r2_out[idxs])
            ) if idxs else float("nan"),
            "mean_rmse": float(
                np.nanmean(rmse_out[idxs])
            ) if idxs else float("nan"),
            "decimate_factor": float(decim),
        }

        results = {
            "params": params_out,
            "r2": r2_out,
            "rmse": rmse_out,
            "channels_fitted": idxs,
        }

        notes = (
            "Fitted double exponential baseline; parameters are stored per "
            "channel. Baseline curves are available in " 
            "derived['double_exp_baseline']."
        )

        return StageOutput(
            signals=new_signals,
            derived={"double_exp_baseline": baseline},
            results=results,
            metrics=metrics,
            notes=notes,
        )


def _summarise_pybaselines_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Make pybaselines params safe to store in state.results:
    - keep scalars
    - keep very small arrays/lists
    - otherwise store a short descriptor
    """
    out: dict[str, Any] = {}

    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
            continue

        if isinstance(v, np.ndarray):
            if v.size <= 16:
                out[k] = np.asarray(v).tolist()
            else:
                out[k] = {"type": "ndarray", "shape": v.shape, "dtype": str(v.dtype)}
            continue

        if isinstance(v, (list, tuple)):
            if len(v) <= 16 and all(
                isinstance(x, (int, float, str, bool)) for x in v
            ):
                out[k] = list(v)
            else:
                out[k] = {"type": type(v).__name__, "len": len(v)}
            continue

        out[k] = repr(v)[:200]

    return out


@dataclass(frozen=True, slots=True)
class PyBaselinesBaseline(UpdateStage):
    """
    Generic baseline estimation using `pybaselines`.

    pybaselines uses a Baseline(x_data=...) object; each algorithm is called as:
        baseline, params = baseline_fitter.<method>(y, **kwargs)

    This stage:
      - computes a baseline per selected channel
      - stores a full (n_signals, n_samples) baseline array in state.derived
      - optionally subtracts baseline from the signal(s)
    """

    name: str = field(default="pybaselines_baseline", init=False)

    method: str = "asls"
    method_kwargs: dict[str, Any] = field(default_factory=dict)

    channels: str | list[str] | None = None

    # x-axis passed to Baseline(x_data=...)
    x_axis: str = "time"  # "time" or "index"

    # where to store the baseline in derived
    baseline_key: str | None = None

    # apply correction?
    subtract: bool = False

    # store full params (can be large); default stores a safe summary
    store_full_params: bool = False

    def _params_for_summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "method_kwargs": self.method_kwargs,
            "channels": self.channels if self.channels is not None else "all",
            "x_axis": self.x_axis,
            "baseline_key": self.baseline_key,
            "subtract": self.subtract,
            "store_full_params": self.store_full_params,
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        try:
            from pybaselines import Baseline
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "pybaselines is required for PyBaselinesBaseline. "
                "Install it with `pip install pybaselines`."
            ) from exc

        idxs = _resolve_channels(state, self.channels)

        if self.x_axis == "time":
            x_data = np.asarray(state.time_seconds, dtype=float)
        elif self.x_axis == "index":
            x_data = np.arange(state.n_samples, dtype=float)
        else:
            raise ValueError("x_axis must be 'time' or 'index'.")

        fitter = Baseline(x_data=x_data)

        if not hasattr(fitter, self.method):
            raise ValueError(
                f"Unknown pybaselines method '{self.method}'. "
                "See pybaselines docs for available algorithms."
            )

        fn = getattr(fitter, self.method)

        # full-shape baseline so downstream code can rely on shape == signals.shape
        baseline = np.full_like(state.signals, np.nan, dtype=float)
        new = state.signals.copy()

        per_channel: dict[str, dict[str, Any]] = {}

        for i in idxs:
            y = np.asarray(state.signals[i], dtype=float)

            b, params = fn(y, **self.method_kwargs)

            # ensure dtype/shape sanity
            b = np.asarray(b, dtype=float)
            if b.shape != y.shape:
                raise ValueError(
                    f"pybaselines returned baseline shape {b.shape}, expected {y.shape}."
                )

            baseline[i] = b
            if self.subtract:
                new[i] = y - b

            per_channel[state.channel_names[i]] = {
                "method": self.method,
                "method_kwargs": dict(self.method_kwargs),
                "params": (params if self.store_full_params else _summarise_pybaselines_params(params)),
            }

        key = (
            self.baseline_key
            if self.baseline_key is not None
            else f"pybaselines_{self.method}"
        )

        notes = (
            f"Computed baseline using pybaselines.{self.method}; "
            f"stored in derived['{key}']."
            + (" Subtracted from signals." if self.subtract else "")
        )

        return StageOutput(
            signals=new if self.subtract else None,
            derived={key: baseline},
            results={
                "baseline_key": key,
                "method": self.method,
                "method_kwargs": dict(self.method_kwargs),
                "channels_fitted": [int(i) for i in idxs],
                "channels": per_channel,
                "x_axis": self.x_axis,
                "subtract": self.subtract,
            },
            notes=notes,
        )

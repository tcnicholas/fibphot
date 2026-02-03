from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..state import PhotometryState
from ..types import FloatArray

ParamOrder = Literal["const", "amp_fast", "amp_slow", "tau_fast", "tau_slow"]


@dataclass(frozen=True, slots=True)
class DoubleExpParams:
    const: float
    amp_fast: float
    amp_slow: float
    tau_fast: float
    tau_slow: float

    @classmethod
    def from_row(cls, row: FloatArray) -> DoubleExpParams:
        r = np.asarray(row, dtype=float).ravel()
        if r.shape[0] != 5:
            raise ValueError(f"Expected 5 params, got shape {r.shape}.")
        return cls(
            const=float(r[0]),
            amp_fast=float(r[1]),
            amp_slow=float(r[2]),
            tau_fast=float(r[3]),
            tau_slow=float(r[4]),
        )

    def f0(self, t: FloatArray) -> FloatArray:
        t = np.asarray(t, dtype=float)
        return (
            self.const
            + self.amp_fast * np.exp(-t / self.tau_fast)
            + self.amp_slow * np.exp(-t / self.tau_slow)
        )


def _last_stage_id(state: PhotometryState, stage_name: str) -> str:
    target = stage_name.lower()
    for rec in reversed(state.summary):
        if rec.name.lower() == target:
            return rec.stage_id
    raise KeyError(f"Stage not found in summary: {stage_name!r}")


def _half_drop_time(t: FloatArray, y: FloatArray) -> float:
    """
    Time at which y has completed half of its total drop from y[0] to y[-1].

    If there is no net drop (y[-1] >= y[0]) returns NaN.
    Uses linear interpolation between samples.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.ndim != 1 or y.ndim != 1 or t.shape[0] != y.shape[0]:
        raise ValueError("t and y must be 1D arrays of the same length.")

    y0 = float(y[0])
    y1 = float(y[-1])
    drop = y0 - y1
    if not np.isfinite(drop) or drop <= 0.0:
        return float("nan")

    target = y0 - 0.5 * drop

    # Find the first index where we are at/below target (assuming decay).
    idx = np.where(y <= target)[0]
    if idx.size == 0:
        return float("nan")

    j = int(idx[0])
    if j == 0:
        return float(t[0])

    # Linear interpolation between (j-1) and j
    t_lo, t_hi = float(t[j - 1]), float(t[j])
    y_lo, y_hi = float(y[j - 1]), float(y[j])

    denom = y_hi - y_lo
    if not np.isfinite(denom) or abs(denom) < 1e-20:
        return float(t_hi)

    frac = (target - y_lo) / denom
    frac = float(np.clip(frac, 0.0, 1.0))
    return t_lo + frac * (t_hi - t_lo)


def _norm_shape_rmse(a: FloatArray, b: FloatArray, eps: float = 1e-12) -> float:
    """
    RMSE between two traces after normalising each by its first sample.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")

    a0 = float(a[0])
    b0 = float(b[0])
    if not (
        np.isfinite(a0) or not np.isfinite(b0) or abs(a0) < eps or abs(b0) < eps
    ):
        return float("nan")

    an = a / a0
    bn = b / b0
    d = an - bn
    return float(np.sqrt(np.nanmean(d * d)))


def photobleaching_summary(
    state: PhotometryState,
    *,
    stage_id: str | None = None,
    stage_name: str = "double_exp_baseline",
    fast_component_threshold: float = 0.05,
    control: str | None = None,
) -> pd.DataFrame:
    """
    Per-channel summary of a fitted double exponential baseline.

    Adds:
      - has_fast_component: whether amp_fast / (amp_fast + amp_slow) exceeds
        `fast_component_threshold`
      - half_drop_time_s: time to complete half of the session-long drop
      - optional control comparisons if `control` is provided:
          * control_percent_drop
          * percent_drop_minus_control
          * tau_slow_ratio_to_control
          * norm_shape_rmse_to_control
    """
    sid = stage_id or _last_stage_id(state, stage_name)
    res = state.results.get(sid)
    if res is None:
        raise KeyError(f"No results found for stage_id={sid!r}.")

    params = np.asarray(res.get("params"), dtype=float)
    r2 = np.asarray(res.get("r2"), dtype=float) if "r2" in res else None
    rmse = np.asarray(res.get("rmse"), dtype=float) if "rmse" in res else None

    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError(f"Expected params shape (n, 5), got {params.shape}.")

    t = np.asarray(state.time_seconds, dtype=float)

    baseline_curves: dict[str, FloatArray] = {}
    rows: list[dict[str, float | str | bool]] = []

    for i, name in enumerate(state.channel_names):
        p = DoubleExpParams.from_row(params[i])

        f = p.f0(t)
        baseline_curves[name] = f

        f_start = float(f[0])
        f_end = float(f[-1])

        amp_total = p.amp_fast + p.amp_slow
        if amp_total != 0.0:
            fast_frac = float(p.amp_fast / amp_total)
            slow_frac = float(p.amp_slow / amp_total)
        else:
            fast_frac = float("nan")
            slow_frac = float("nan")

        has_fast = bool(
            np.isfinite(fast_frac) and fast_frac > fast_component_threshold
        )

        if f_start != 0.0 and np.isfinite(f_start) and np.isfinite(f_end):
            drop_frac = float((f_start - f_end) / f_start)
            percent_drop = 100.0 * drop_frac
        else:
            percent_drop = float("nan")

        row: dict[str, float | str | bool] = {
            "channel": name,
            "tau_fast_s": p.tau_fast,
            "tau_slow_s": p.tau_slow,
            "fast_amp_frac": fast_frac,
            "slow_amp_frac": slow_frac,
            "has_fast_component": has_fast,
            "half_drop_time_s": _half_drop_time(t, f),
            "percent_drop": percent_drop,
            "const": p.const,
            "amp_fast": p.amp_fast,
            "amp_slow": p.amp_slow,
            "f0_start": f_start,
            "f0_end": f_end,
        }
        if r2 is not None and i < r2.shape[0]:
            row["r2"] = float(r2[i])
        if rmse is not None and i < rmse.shape[0]:
            row["rmse"] = float(rmse[i])

        rows.append(row)

    df = pd.DataFrame(rows)

    if control is not None:
        ctl = control.lower()
        if ctl not in baseline_curves:
            raise KeyError(
                f"Control channel {control!r} not found. "
                f"Available: {state.channel_names}"
            )

        ctl_row = df.loc[df["channel"] == ctl]
        if ctl_row.empty:
            raise KeyError(f"Control channel {control!r} not found in summary.")
        ctl_percent_drop = float(ctl_row["percent_drop"].iloc[0])
        ctl_tau_slow = float(ctl_row["tau_slow_s"].iloc[0])
        ctl_curve = baseline_curves[ctl]

        df["control_percent_drop"] = ctl_percent_drop
        df["percent_drop_minus_control"] = df["percent_drop"] - ctl_percent_drop
        df["tau_slow_ratio_to_control"] = df["tau_slow_s"] / ctl_tau_slow

        rmses: list[float] = []
        for ch in df["channel"].tolist():
            rmses.append(_norm_shape_rmse(baseline_curves[ch], ctl_curve))
        df["norm_shape_rmse_to_control"] = rmses

    preferred = [
        "channel",
        "tau_fast_s",
        "tau_slow_s",
        "fast_amp_frac",
        "slow_amp_frac",
        "has_fast_component",
        "half_drop_time_s",
        "percent_drop",
        "r2",
        "rmse",
    ]
    if control is not None:
        preferred += [
            "control_percent_drop",
            "percent_drop_minus_control",
            "tau_slow_ratio_to_control",
            "norm_shape_rmse_to_control",
        ]
    preferred += ["const", "amp_fast", "amp_slow", "f0_start", "f0_end"]

    cols = [c for c in preferred if c in df.columns]
    return df[cols].sort_values("channel")


def get_baseline_trace(
    state: PhotometryState,
    *,
    baseline_key: str = "double_exp_baseline",
    channel: str,
    normalise_to_start: bool = False,
    eps: float = 1e-12,
) -> FloatArray:
    """
    Fetch a baseline trace from state.derived and optionally normalise to start.
    """
    if baseline_key not in state.derived:
        raise KeyError(
            f"derived['{baseline_key}'] not found. "
            "Run the baseline stage first."
        )
    base = np.asarray(state.derived[baseline_key], dtype=float)
    if base.shape != state.signals.shape:
        raise ValueError(
            f"derived['{baseline_key}'] has shape {base.shape}, "
            f"expected {state.signals.shape}."
        )

    i = state.idx(channel)
    y = base[i].copy()

    if normalise_to_start:
        d0 = float(y[0])
        if not np.isfinite(d0) or abs(d0) < eps:
            y[:] = np.nan
        else:
            y = y / d0

    return y

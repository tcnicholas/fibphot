from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..fit.regression import fit_irls
from ..state import PhotometryState
from ..types import FloatArray

Loss = Literal["tukey", "huber"]
PlotView = Literal["current", "raw", "before_stage", "after_stage"]


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """How to select x/y from a state for the sweep."""

    control: str = "iso"
    channel: str = "gcamp"
    view: PlotView = "current"
    stage_name: str | None = None
    stage_id: str | None = None
    occurrence: int = -1


def _snapshot_at(state: PhotometryState, state_index: int) -> FloatArray:
    """
    Return the signals snapshot after `state_index` stages.

    For k applied stages:
      - snapshot 0: raw (after 0 stages)
      - snapshot j: after j stages
      - snapshot k: current (after k stages)
    """
    k = len(state.summary)
    if state_index < 0 or state_index > k:
        raise ValueError(f"state_index must be in [0, {k}], got {state_index}.")

    if k == 0:
        return state.signals

    if state_index < k:
        return state.history[state_index]

    return state.signals


def _find_stage_index(
    state: PhotometryState,
    *,
    stage_name: str | None,
    stage_id: str | None,
    occurrence: int,
) -> int:
    if stage_name is None and stage_id is None:
        raise ValueError(
            "Provide stage_name or stage_id for stage-based views."
        )

    matches: list[int] = []
    for i, rec in enumerate(state.summary):
        if (stage_id is not None and rec.stage_id == stage_id) or (
            stage_name is not None and rec.name.lower() == stage_name.lower()
        ):
            matches.append(i)

    if not matches:
        key = stage_id if stage_id is not None else stage_name
        raise KeyError(f"Stage not found in summary: {key!r}")

    try:
        return matches[occurrence]
    except IndexError as exc:
        raise IndexError(
            f"Stage occurrence {occurrence} is out of range. "
            f"Found {len(matches)} occurrence(s)."
        ) from exc


def _signals_for_view(
    state: PhotometryState,
    *,
    view: PlotView,
    stage_name: str | None,
    stage_id: str | None,
    occurrence: int,
) -> FloatArray:
    if view == "current":
        return state.signals

    if view == "raw":
        return _snapshot_at(state, 0)

    si = _find_stage_index(
        state,
        stage_name=stage_name,
        stage_id=stage_id,
        occurrence=occurrence,
    )
    if view == "before_stage":
        return _snapshot_at(state, si)
    if view == "after_stage":
        return _snapshot_at(state, si + 1)

    raise ValueError(f"Unknown view: {view!r}")


def _weight_stats(w: FloatArray | None) -> dict[str, float]:
    if w is None:
        return {
            "w_mean": float("nan"),
            "w_median": float("nan"),
            "w_min": float("nan"),
            "w_p01": float("nan"),
            "w_p05": float("nan"),
            "w_zero_frac": float("nan"),
            "w_lt_01_frac": float("nan"),
        }

    ww = np.asarray(w, dtype=float)
    ww = ww[np.isfinite(ww)]
    if ww.size == 0:
        return {
            "w_mean": float("nan"),
            "w_median": float("nan"),
            "w_min": float("nan"),
            "w_p01": float("nan"),
            "w_p05": float("nan"),
            "w_zero_frac": float("nan"),
            "w_lt_01_frac": float("nan"),
        }

    return {
        "w_mean": float(np.mean(ww)),
        "w_median": float(np.median(ww)),
        "w_min": float(np.min(ww)),
        "w_p01": float(np.quantile(ww, 0.01)),
        "w_p05": float(np.quantile(ww, 0.05)),
        "w_zero_frac": float(np.mean(ww <= 0.0)),
        "w_lt_01_frac": float(np.mean(ww < 0.1)),
    }


def irls_tuning_sweep_xy(
    x: FloatArray,
    y: FloatArray,
    *,
    tuning_constants: list[float] | FloatArray,
    loss: Loss = "tukey",
    include_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
    store_weights: bool = True,
) -> pd.DataFrame:
    """
    Sweep IRLS tuning constants for y ~ a + b x.

    Returns a DataFrame with fit parameters and robustness diagnostics.
    """
    tc = np.asarray(tuning_constants, dtype=float)
    if tc.ndim != 1 or tc.size == 0:
        raise ValueError("tuning_constants must be a non-empty 1D sequence.")
    if np.any(~np.isfinite(tc)) or np.any(tc <= 0.0):
        raise ValueError("All tuning_constants must be finite and > 0.")

    rows: list[dict[str, float]] = []
    for c in tc:
        fit = fit_irls(
            x,
            y,
            include_intercept=include_intercept,
            loss=loss,
            tuning_constant=float(c),
            max_iter=max_iter,
            tol=tol,
            store_weights=store_weights,
        )
        row: dict[str, float] = {
            "tuning_constant": float(c),
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r2": float(fit.r2),
            "n_iter": float(fit.n_iter)
            if fit.n_iter is not None
            else float("nan"),
            "scale": float(fit.scale)
            if fit.scale is not None
            else float("nan"),
        }
        row.update(_weight_stats(fit.weights))
        rows.append(row)

    df = (
        pd.DataFrame(rows).sort_values("tuning_constant").reset_index(drop=True)
    )
    df["loss"] = loss
    return df


def irls_tuning_sweep(
    state: PhotometryState,
    *,
    tuning_constants: list[float] | FloatArray,
    loss: Loss = "tukey",
    include_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
    store_weights: bool = True,
    spec: SweepSpec | None = None,
) -> pd.DataFrame:
    """
    Sweep IRLS tuning constants using x/y taken from a PhotometryState.

    Tip:
      - If you want the *pre-regression* signals for sensitivity analysis, use:
            spec=SweepSpec(view="before_stage", stage_name="isosbestic_regression")
        on the *post* state that contains history.
      - If you're sweeping before you've applied regression, just use view="current".
    """
    if spec is None:
        spec = SweepSpec()

    sig = _signals_for_view(
        state,
        view=spec.view,
        stage_name=spec.stage_name,
        stage_id=spec.stage_id,
        occurrence=spec.occurrence,
    )

    x = sig[state.idx(spec.control)]
    y = sig[state.idx(spec.channel)]

    df = irls_tuning_sweep_xy(
        x,
        y,
        tuning_constants=tuning_constants,
        loss=loss,
        include_intercept=include_intercept,
        max_iter=max_iter,
        tol=tol,
        store_weights=store_weights,
    )
    df["control"] = spec.control.lower()
    df["channel"] = spec.channel.lower()
    df["view"] = spec.view
    df["stage_name"] = spec.stage_name if spec.stage_name is not None else ""
    return df


def plot_irls_tuning_sweep(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (6.0, 3.0),
    dpi: int = 150,
    font_size: int = 8,
    show_weights: bool = True,
) -> tuple[Figure, tuple[Any, Any, Any, Axes | None]]:
    """
    Plot slope/intercept sensitivity vs tuning_constant.

    Returns (fig, (ax_params, ax_diag)).
    """
    required = {"tuning_constant", "slope", "intercept", "r2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {sorted(missing)}"
        )

    x = np.asarray(df["tuning_constant"], dtype=float)

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    def style(ax: Axes) -> None:
        ax.tick_params(labelsize=font_size)
        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    style(ax_top)
    style(ax_bot)

    # --- Top panel: slope + intercept on twinx ---
    ax_top_r = ax_top.twinx()
    ax_top_r.tick_params(labelsize=font_size)
    ax_top_r.yaxis.label.set_fontsize(font_size)
    ax_top_r.spines["top"].set_visible(False)

    l1 = ax_top.plot(x, df["slope"], linewidth=1.2, label="slope")
    l2 = ax_top_r.plot(
        x,
        df["intercept"],
        linewidth=1.2,
        linestyle="--",
        label="intercept",
    )

    ax_top.set_ylabel("slope")
    ax_top_r.set_ylabel("intercept")

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax_top.legend(lines, labels, frameon=False, fontsize=font_size)

    # --- Bottom panel: R² + optional weights on twinx ---
    ax_bot_r: Axes | None = None
    l3 = ax_bot.plot(x, df["r2"], linewidth=1.2, label="R²")
    ax_bot.set_ylabel("R²")
    ax_bot.set_xlabel("tuning constant")

    lines2 = l3
    labels2 = [ln.get_label() for ln in l3]

    if show_weights and "w_zero_frac" in df.columns:
        ax_bot_r = ax_bot.twinx()
        assert ax_bot_r is not None
        ax_bot_r.tick_params(labelsize=font_size)
        ax_bot_r.yaxis.label.set_fontsize(font_size)
        ax_bot_r.spines["top"].set_visible(False)

        l4 = ax_bot_r.plot(
            x,
            df["w_zero_frac"],
            linewidth=1.0,
            linestyle=":",
            label="zero-weight frac",
        )
        ax_bot_r.set_ylabel("zero-weight frac")

        lines2 = l3 + l4
        labels2 = [ln.get_label() for ln in lines2]

    ax_bot.legend(lines2, labels2, frameon=False, fontsize=font_size)

    return fig, (ax_top, ax_bot, ax_top_r, ax_bot_r)

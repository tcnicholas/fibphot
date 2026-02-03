from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..state import PhotometryState
from .report import AnalysisResult


def plot_auc_result(
    state: PhotometryState,
    result: AnalysisResult,
    *,
    ax=None,
    label: str | None = None,
    signal_alpha: float = 0.9,
    window_alpha: float = 0.08,
    fill_alpha: float = 0.30,
    linewidth: float = 1.2,
    fontsize: int = 8,
    show_metrics: bool = True,
    colour_signal: str = "#1f77b4",   # vivid blue
    colour_baseline: str = "#444444", # dark grey
    colour_fill: str = "#ff7f0e",     # vivid orange
):
    """
    Visualise an AUC AnalysisResult:
      - plots full trace
      - shows baseline reference line
      - shades the analysis window
      - fills the integrated area (using stored window contrib)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    else:
        fig = ax.figure

    chan = result.channel
    t = np.asarray(state.time_seconds, float)
    y = np.asarray(state.channel(chan), float)

    lab = label or chan
    ax.plot(t, y, linewidth=linewidth, alpha=signal_alpha, label=lab, color=colour_signal)

    b = float(result.metrics.get("baseline", np.nan))
    if np.isfinite(b):
        ax.axhline(b, linewidth=1.0, alpha=0.85, linestyle="--", color=colour_baseline)

    # Window display
    t0 = result.metrics.get("t0", None)
    t1 = result.metrics.get("t1", None)
    if t0 is not None and t1 is not None and np.isfinite(t0) and np.isfinite(t1):
        ax.axvspan(float(t0), float(t1), alpha=window_alpha, color="grey")

    # Fill based on stored arrays
    tt = np.asarray(result.arrays.get("t_window", np.array([])), float)
    contrib = np.asarray(result.arrays.get("contrib", np.array([])), float)

    if tt.size >= 2 and contrib.size == tt.size and np.isfinite(b):
        ax.fill_between(tt, b, b + contrib, alpha=fill_alpha, color=colour_fill, label="AUC area")

    ax.set_xlabel("time (s)", fontsize=fontsize)
    ax.set_ylabel(chan, fontsize=fontsize)

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    if show_metrics:
        auc = result.metrics.get("auc", np.nan)
        baseline = result.metrics.get("baseline", np.nan)
        txt = f"AUC: {auc:.3g}\nBaseline: {baseline:.3g}"
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            color="#222222",
        )

    ax.legend(frameon=False, fontsize=fontsize)

    return fig, ax

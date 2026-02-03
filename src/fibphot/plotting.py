from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from .state import PhotometryState
from .types import FloatArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

PlotView = Literal["current", "raw", "before_stage", "after_stage"]
PlotMode = Literal["overlay", "stacked"]


@dataclass(frozen=True, slots=True)
class PlotTheme:
    figsize: tuple[float, float] = (6.0, 3.0)
    dpi: int = 150

    label_size: int = 8
    tick_size: int = 8
    legend_size: int = 8
    title_size: int = 8

    linewidth: float = 1.1
    alpha: float = 1.0

    # Vibrant, colour-blind friendly
    signal_colour: str = "#0072B2"       # blue
    control_colour: str = "#000000"      # black
    baseline_colour: str = "#E69F00"     # orange (baseline fit)
    fit_colour: str = "#009E73"          # green (motion/control fit)
    difference_colour: str = "#D55E00"   # vermillion
    accent_purple: str = "#CC79A7"       # purple

    cycle: tuple[str, ...] = (
        "#0072B2",  # blue
        "#009E73",  # green
        "#D55E00",  # vermillion
        "#E69F00",  # orange
        "#CC79A7",  # purple
        "#56B4E9",  # light blue
    )


def set_plot_defaults() -> None:
    """Set matplotlib rcParams for consistent plotting style."""

    theme = PlotTheme()

    plt.rcParams.update(
        {
            "figure.dpi": theme.dpi,
            "savefig.dpi": theme.dpi,
            "font.size": theme.label_size,
            "axes.titlesize": theme.title_size,
            "axes.labelsize": theme.label_size,
            "xtick.labelsize": theme.tick_size,
            "ytick.labelsize": theme.tick_size,
        }
    )

def _apply_theme(ax: Axes, theme: PlotTheme) -> None:
    ax.tick_params(labelsize=theme.tick_size)
    ax.xaxis.label.set_fontsize(theme.label_size)
    ax.yaxis.label.set_fontsize(theme.label_size)
    ax.title.set_fontsize(theme.title_size)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _new_axes(ax: Axes | None, theme: PlotTheme) -> tuple[Figure | SubFigure, Axes]:
    if ax is not None:
        fig = ax.figure
        _apply_theme(ax, theme)
        return fig, ax

    fig, ax = plt.subplots(
        figsize=theme.figsize,
        dpi=theme.dpi,
        constrained_layout=True,
    )
    _apply_theme(ax, theme)
    return fig, ax


def _snapshot_at(state: PhotometryState, state_index: int) -> FloatArray:
    """
    Return the signals snapshot after `state_index` stages.

    For k applied stages:
      - snapshot 0: raw (after 0 stages)
      - snapshot j: after j stages
      - snapshot k: current (after k stages)

    With your state design, history has length k and stores snapshots for
    state_index in [0, k-1]; current is state.signals.
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
    """Return 0-based index into state.summary for the requested stage."""
    if stage_name is None and stage_id is None:
        raise ValueError("Provide stage_name or stage_id.")

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
    """Select which signals snapshot to plot."""
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


def plot_current(
    state: PhotometryState,
    *,
    signal: str,
    control: str | None = None,
    baseline_key: str | None = None,
    motion_fit_key: str | None = None,
    overlay_offset: dict[str, float] | None = None,
    view: PlotView = "current",
    stage_name: str | None = None,
    stage_id: str | None = None,
    occurrence: int = -1,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_legend: bool = True,
    theme: PlotTheme | None = None,
    ax: Axes | None = None,
    label: str | None = None,
    colour: str | None = None,
    linestyle: str | None = None,
    alpha: float | None = None,
    linewidth: float | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """
    Plot a single signal from a chosen snapshot, optionally with control and
    derived overlays.

    Typical use after `DoubleExpBaseline(subtract=True)`:

        state.plot(
            signal="gcamp",
            baseline_key="double_exp_baseline",
            view="before_stage",
            stage_name="double_exp_baseline",
        )

    baseline_key:
        A derived array with shape == state.signals.shape, e.g.
        "double_exp_baseline". Only the `signal` channel is drawn.

    motion_fit_key:
        A derived array with shape == state.signals.shape, e.g. "motion_fit".

    overlay_offset:
        Optional per-overlay constant offsets applied when plotting overlays,
        keyed by the overlay name. For example:

            overlay_offset={"motion_fit": -0.01}

        Offsets are applied only to overlays (not the main signal/control).
    """
    theme = theme or PlotTheme()
    fig, ax = _new_axes(ax, theme)

    offsets = overlay_offset or {}

    def _offset_for(key: str) -> float:
        v = offsets.get(key, 0.0)
        return float(v)

    t = state.time_seconds
    sig = _signals_for_view(
        state,
        view=view,
        stage_name=stage_name,
        stage_id=stage_id,
        occurrence=occurrence,
    )

    i_sig = state.idx(signal)
    y = sig[i_sig]

    ax.plot(
        t,
        y,
        label=label or f"{signal.lower()} ({view})",
        color=colour or theme.signal_colour,
        linewidth=theme.linewidth if linewidth is None else float(linewidth),
        alpha=theme.alpha if alpha is None else float(alpha),
        linestyle=linestyle or "-",
    )

    if control is not None:
        i_ctl = state.idx(control)
        c = sig[i_ctl]
        ax.plot(
            t,
            c,
            label=f"{control.lower()} ({view})",
            color=theme.control_colour,
            linewidth=max(0.9, theme.linewidth * 0.9),
            alpha=0.75,
        )

    if baseline_key is not None and baseline_key in state.derived:
        base = np.asarray(state.derived[baseline_key], dtype=float)
        if base.shape != state.signals.shape:
            raise ValueError(
                f"derived['{baseline_key}'] has shape {base.shape}, "
                f"expected {state.signals.shape}."
            )

        off = _offset_for(baseline_key)
        ax.plot(
            t,
            base[i_sig] + off,
            label=baseline_key if off==0.0 else f"{baseline_key} (offset)",
            color=theme.baseline_colour,
            linewidth=max(1.6, theme.linewidth * 1.6),
            alpha=1.0,
            linestyle="--",
        )

    if motion_fit_key is not None and motion_fit_key in state.derived:
        mf = np.asarray(state.derived[motion_fit_key], dtype=float)
        if mf.shape != state.signals.shape:
            raise ValueError(
                f"derived['{motion_fit_key}'] has shape {mf.shape}, "
                f"expected {state.signals.shape}."
            )

        off = _offset_for(motion_fit_key)
        ax.plot(
            t,
            mf[i_sig] + off,
            label=motion_fit_key if off==0.0 else f"{motion_fit_key} (offset)",
            color=theme.fit_colour,
            linewidth=max(1.4, theme.linewidth * 1.4),
            alpha=0.95,
            linestyle=":",
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel(signal.lower())

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_legend:
        ax.legend(frameon=False, fontsize=theme.legend_size)

    return fig, ax



def _label_for_state_index(
    state: PhotometryState,
    state_index: int,
    *,
    include_index: bool,
    include_current_tag: bool,
) -> str:
    k = len(state.summary)
    base = "raw" if state_index <= 0 else state.summary[state_index - 1].name

    parts: list[str] = []
    if include_index:
        parts.append(f"{state_index:02d}")
    parts.append(base)

    label = " - ".join(parts)
    if include_current_tag and state_index == k and k > 0:
        label = f"{label} (current)"
    return label


def plot_history(
    state: PhotometryState,
    channel: str,
    *,
    stage_name: str | None = None,
    stage_id: str | None = None,
    occurrence: int = -1,
    around: bool = False,
    plot_difference: bool = False,
    difference_label: str | None = None,
    n_recent: int | None = None,
    include_raw: bool = True,
    include_current: bool = True,
    include_index: bool = True,
    include_current_tag: bool = True,
    mode: PlotMode = "overlay",
    theme: PlotTheme | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure | SubFigure, Axes | tuple[Axes, Axes]]:
    """
    Plot one channel across the saved history (and optionally current).

    If around=True, plot only the snapshot immediately before and immediately
    after a chosen stage.
    """
    theme = theme or PlotTheme()
    ch_i = state.idx(channel)
    t = state.time_seconds
    k = len(state.summary)

    if around:
        if k == 0:
            raise ValueError("No stages have been applied; cannot plot around a stage.")

        si = _find_stage_index(
            state,
            stage_name=stage_name,
            stage_id=stage_id,
            occurrence=occurrence,
        )
        stage = state.summary[si]
        before_sig = _snapshot_at(state, si)
        after_sig = _snapshot_at(state, si + 1)

        y_before = before_sig[ch_i]
        y_after = after_sig[ch_i]
        y_diff = y_after - y_before

        if plot_difference:
            if ax is not None:
                raise ValueError(
                    "plot_difference=True requires ax=None so the function can "
                    "create a two-row figure."
                )

            fig, (ax0, ax1) = plt.subplots(
                nrows=2,
                sharex=True,
                figsize=theme.figsize,
                dpi=theme.dpi,
                constrained_layout=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )
            _apply_theme(ax0, theme)
            _apply_theme(ax1, theme)

            ax0.plot(
                t,
                y_before,
                label=f"before {stage.name}",
                linewidth=theme.linewidth,
                alpha=theme.alpha,
                color=theme.control_colour,
            )
            ax0.plot(
                t,
                y_after,
                label=f"after {stage.name}",
                linewidth=theme.linewidth,
                alpha=theme.alpha,
                color=theme.signal_colour,
            )
            ax0.legend(frameon=False, fontsize=theme.legend_size)

            ax1.plot(
                t,
                y_diff,
                linewidth=theme.linewidth,
                alpha=theme.alpha,
                color=theme.difference_colour,
            )
            ax1.axhline(0.0, linewidth=1.0, color="k")

            ax0.set_ylabel(channel.lower())
            ax1.set_ylabel("difference")
            ax1.set_xlabel("time (s)")
            ax0.set_title(title or f"{channel.lower()} — around {stage.name}")

            return fig, (ax0, ax1)

        fig, ax0 = _new_axes(ax, theme)

        if mode == "overlay":
            ax0.plot(
                t,
                y_before,
                label=f"before {stage.name}",
                linewidth=theme.linewidth,
                alpha=theme.alpha,
                color=theme.control_colour,
            )
            ax0.plot(
                t,
                y_after,
                label=f"after {stage.name}",
                linewidth=theme.linewidth,
                alpha=theme.alpha,
                color=theme.signal_colour,
            )
            ax0.legend(frameon=False, fontsize=theme.legend_size)

        elif mode == "stacked":
            base_scale = float(np.nanpercentile(np.abs(y_after), 95))
            step = base_scale if base_scale > 0 else 1.0

            ax0.plot(
                t,
                y_before,
                label=f"before {stage.name}",
                linewidth=theme.linewidth,
                alpha=theme.alpha,
                color=theme.control_colour,
            )
            ax0.plot(
                t,
                y_after + step,
                label=f"after {stage.name}",
                linewidth=theme.linewidth,
                alpha=theme.alpha,
                color=theme.signal_colour,
            )
            ax0.legend(frameon=False, fontsize=theme.legend_size)

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        ax0.set_xlabel("time (s)")
        ax0.set_ylabel(channel.lower())
        ax0.set_title(title or f"{channel.lower()} — around {stage.name}")

        return fig, ax0

    # ---- default behaviour: plot multiple snapshots ----

    snapshots: list[tuple[int, FloatArray]] = []
    h = int(state.history.shape[0])

    if h == 0:
        snapshots.append((0, state.signals))
    else:
        for j in range(h):
            snapshots.append((j, state.history[j]))
        snapshots.append((h, state.signals))

    filtered: list[tuple[int, FloatArray]] = []
    for state_index, sig in snapshots:
        if state_index == 0 and not include_raw and h > 0:
            continue
        if state_index == k and not include_current:
            continue
        filtered.append((state_index, sig))

    if not filtered:
        raise ValueError(
            "No snapshots selected to plot (check include_* flags)."
        )

    if n_recent is not None:
        if n_recent < 1:
            raise ValueError("n_recent must be >= 1.")
        filtered = filtered[-n_recent:]

    fig, ax0 = _new_axes(ax, theme)
    with contextlib.suppress(Exception):
        ax0.set_prop_cycle(color=list(theme.cycle))

    if mode == "overlay":
        for state_index, sig in filtered:
            y = sig[ch_i]
            label = _label_for_state_index(
                state,
                state_index,
                include_index=include_index,
                include_current_tag=include_current_tag,
            )
            ax0.plot(
                t,
                y,
                label=label,
                linewidth=theme.linewidth,
                alpha=theme.alpha,
            )
        ax0.legend(frameon=False, fontsize=theme.legend_size)

    elif mode == "stacked":
        offset = 0.0
        base_scale = float(np.nanpercentile(np.abs(state.signals[ch_i]), 95))
        step = base_scale if base_scale > 0 else 1.0

        for state_index, sig in filtered:
            y = sig[ch_i]
            label = _label_for_state_index(
                state,
                state_index,
                include_index=include_index,
                include_current_tag=include_current_tag,
            )
            ax0.plot(
                t,
                y + offset,
                label=label,
                linewidth=theme.linewidth,
                alpha=theme.alpha,
            )
            offset += step

        ax0.legend(frameon=False, fontsize=theme.legend_size)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    ax0.set_xlabel("time (s)")
    ax0.set_ylabel(channel.lower())
    if title is not None:
        ax0.set_title(title)

    return fig, ax0

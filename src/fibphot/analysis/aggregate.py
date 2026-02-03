from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..state import PhotometryState
from ..types import FloatArray

AlignMode = Literal["intersection", "union"]
TimeRef = Literal["absolute", "start"]
InterpKind = Literal["linear", "nearest"]


@dataclass(frozen=True, slots=True)
class AlignedSignals:
    """
    Container for a time-aligned stack of signals.

    aligned has shape (n_states, n_channels, n_time).
    """

    time_seconds: FloatArray
    aligned: FloatArray
    channel_names: tuple[str, ...]
    subjects: tuple[str | None, ...]
    align_mode: AlignMode
    time_ref: TimeRef
    dt: float


def _as_relative_time(t: FloatArray) -> FloatArray:
    t = np.asarray(t, dtype=float)
    return t - float(t[0])


def _infer_dt(states: Sequence[PhotometryState]) -> float:
    dts: list[float] = []
    for s in states:
        dt = np.diff(np.asarray(s.time_seconds, dtype=float))
        if dt.size == 0:
            continue
        dts.append(float(np.nanmedian(dt)))
    if not dts:
        raise ValueError(
            "Cannot infer dt from empty or degenerate time arrays."
        )
    return float(np.nanmedian(np.asarray(dts, dtype=float)))


def _time_window(
    times: list[FloatArray],
    mode: AlignMode,
) -> tuple[float, float]:
    starts = [float(t[0]) for t in times]
    ends = [float(t[-1]) for t in times]

    if mode == "intersection":
        t0 = max(starts)
        t1 = min(ends)
    elif mode == "union":
        t0 = min(starts)
        t1 = max(ends)
    else:
        raise ValueError(f"Unknown align mode: {mode!r}")

    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError(
            f"Invalid common window from mode={mode!r}: ({t0}, {t1})."
        )
    return t0, t1


def _make_grid(t0: float, t1: float, dt: float) -> FloatArray:
    n = int(np.floor((t1 - t0) / dt)) + 1
    if n < 2:
        raise ValueError("Common time grid would have <2 points.")
    return t0 + dt * np.arange(n, dtype=float)


def _interp_1d(
    x: FloatArray,
    y: FloatArray,
    x_new: FloatArray,
    *,
    kind: InterpKind,
    fill: float,
) -> FloatArray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x0 = x[mask]
    y0 = y[mask]
    if x0.size < 2:
        return np.full_like(x_new, fill, dtype=float)

    order = np.argsort(x0)
    x0 = x0[order]
    y0 = y0[order]

    if kind == "linear":
        return np.interp(x_new, x0, y0, left=fill, right=fill).astype(float)

    if kind == "nearest":
        idx = np.searchsorted(x0, x_new, side="left")
        idx = np.clip(idx, 0, x0.size - 1)
        left = np.clip(idx - 1, 0, x0.size - 1)

        choose_left = np.abs(x_new - x0[left]) <= np.abs(x_new - x0[idx])
        out = np.where(choose_left, y0[left], y0[idx]).astype(float)

        out[(x_new < x0[0]) | (x_new > x0[-1])] = fill
        return out

    raise ValueError(f"Unknown interpolation kind: {kind!r}")


def align_collection_signals(
    states: Sequence[PhotometryState],
    *,
    channels: Sequence[str] | None = None,
    align: AlignMode = "intersection",
    time_ref: TimeRef = "start",
    dt: float | None = None,
    target_fs: float | None = None,
    interpolation: InterpKind = "linear",
    fill: float = float("nan"),
) -> AlignedSignals:
    """
    Align a set of states to a common time axis via interpolation.

    align="intersection":
        Uses only the time interval present in all states (truncate behaviour).
    align="union":
        Uses the full time span across states and fills missing regions (pad behaviour).

    time_ref="start":
        Treats each state's time as relative to its own start time.
    time_ref="absolute":
        Uses each state's original time_seconds values.
    """
    if not states:
        raise ValueError("No states provided.")

    if channels is None:
        channel_names = states[0].channel_names
        idxs = list(range(states[0].n_signals))
    else:
        channel_names = tuple(c.lower() for c in channels)
        idxs = [states[0].idx(c) for c in channel_names]

    for s in states[1:]:
        for c in channel_names:
            _ = s.idx(c)

    if target_fs is not None:
        if target_fs <= 0:
            raise ValueError("target_fs must be > 0.")
        dt_use = 1.0 / float(target_fs)
    else:
        dt_use = float(dt) if dt is not None else _infer_dt(states)

    times: list[FloatArray] = []
    for s in states:
        t = np.asarray(s.time_seconds, dtype=float)
        if time_ref == "start":
            t = _as_relative_time(t)
        times.append(t)

    t0, t1 = _time_window(times, align)
    grid = _make_grid(t0, t1, dt_use)

    aligned = np.full(
        (len(states), len(idxs), grid.size),
        fill,
        dtype=float,
    )

    for si, s in enumerate(states):
        t = times[si]
        for cj, idx in enumerate(idxs):
            y = np.asarray(s.signals[idx], dtype=float)
            aligned[si, cj] = _interp_1d(
                t,
                y,
                grid,
                kind=interpolation,
                fill=fill,
            )

    subjects: tuple[str | None, ...] = tuple(
        getattr(s, "subject", None) for s in states
    )

    return AlignedSignals(
        time_seconds=grid,
        aligned=aligned,
        channel_names=tuple(channel_names),
        subjects=subjects,
        align_mode=align,
        time_ref=time_ref,
        dt=dt_use,
    )


def mean_aligned(
    aligned: AlignedSignals,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Compute mean, std, sem, and n (per channel/time) from an aligned stack.
    """
    x = np.asarray(aligned.aligned, dtype=float)

    n = np.sum(np.isfinite(x), axis=0).astype(float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)

    sem = std / np.sqrt(np.where(n <= 0, np.nan, n))
    return mean, std, sem, n


def mean_state_from_aligned(
    aligned: AlignedSignals,
    *,
    name: str = "group_mean",
) -> PhotometryState:
    """
    Return a PhotometryState whose signals are the group mean.

    Additional statistics are stored in derived:
        - derived["group_std"]
        - derived["group_sem"]
        - derived["group_n"]
    """
    mean, std, sem, n = mean_aligned(aligned)

    return PhotometryState(
        time_seconds=aligned.time_seconds,
        signals=mean,
        channel_names=aligned.channel_names,
        derived={
            "group_std": std,
            "group_sem": sem,
            "group_n": n,
        },
        metadata={
            "kind": name,
            "align_mode": aligned.align_mode,
            "time_ref": aligned.time_ref,
            "dt": aligned.dt,
            "subjects": list(aligned.subjects),
        },
    )

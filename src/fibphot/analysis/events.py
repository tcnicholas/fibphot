from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..state import PhotometryState
from ..types import FloatArray


@dataclass(frozen=True, slots=True)
class EventAlignedTraces:
    time_relative_s: FloatArray
    data: FloatArray  # n_events x n_channels x n_time
    channel_names: tuple[str, ...]
    event_times_s: FloatArray
    event_metadata: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def n_events(self) -> int:
        return int(self.data.shape[0])

    def mean(self) -> FloatArray:
        return np.nanmean(self.data, axis=0)

    def sem(self) -> FloatArray:
        n = np.sum(np.isfinite(self.data), axis=0)
        sd = np.nanstd(self.data, axis=0, ddof=1)
        return sd / np.sqrt(np.maximum(n, 1))

    def to_dataframe(self):
        import pandas as pd
        rows: list[dict[str, Any]] = []
        for ei, event_t in enumerate(self.event_times_s):
            meta = self.event_metadata[ei] if ei < len(self.event_metadata) else {}
            for ci, channel in enumerate(self.channel_names):
                for ti, trel in enumerate(self.time_relative_s):
                    rows.append({"event_i": ei, "event_time_s": event_t, "channel": channel, "time_relative_s": trel, "value": self.data[ei, ci, ti], **meta})
        return pd.DataFrame(rows)


def _as_event_times(events_s: Sequence[float] | Mapping[str, Sequence[float]]) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    if isinstance(events_s, Mapping):
        times: list[float] = []
        meta: list[dict[str, Any]] = []
        for label, vals in events_s.items():
            for v in vals:
                times.append(float(v))
                meta.append({"event_label": str(label)})
        order = np.argsort(times)
        return np.asarray(times, dtype=float)[order], tuple(meta[int(i)] for i in order)
    arr = np.asarray(list(events_s), dtype=float)
    return arr, tuple({} for _ in arr)


def align_to_events(
    state: PhotometryState,
    events_s: Sequence[float] | Mapping[str, Sequence[float]],
    *,
    t_before: float = 5.0,
    t_after: float = 10.0,
    channels: Sequence[str] | None = None,
    dt: float | None = None,
    target_fs: float | None = None,
    fill: float = float("nan"),
) -> EventAlignedTraces:
    if t_before < 0 or t_after < 0:
        raise ValueError("t_before and t_after must be non-negative.")
    event_times, meta = _as_event_times(events_s)
    event_times = event_times[np.isfinite(event_times)]
    if channels is None:
        channel_names = state.channel_names
        idxs = list(range(state.n_signals))
    else:
        channel_names = tuple(str(c).lower() for c in channels)
        idxs = [state.idx(c) for c in channel_names]
    if target_fs is not None:
        if target_fs <= 0:
            raise ValueError("target_fs must be > 0.")
        step = 1.0 / float(target_fs)
    elif dt is not None:
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        step = float(dt)
    else:
        step = 1.0 / state.sampling_rate
    n = int(np.floor((t_before + t_after) / step)) + 1
    rel = -float(t_before) + step * np.arange(n, dtype=float)
    data = np.full((event_times.size, len(idxs), rel.size), fill, dtype=float)
    t = np.asarray(state.time_seconds, dtype=float)
    for ei, ev in enumerate(event_times):
        target = ev + rel
        for ci, idx in enumerate(idxs):
            y = np.asarray(state.signals[idx], dtype=float)
            mask = np.isfinite(t) & np.isfinite(y)
            if np.sum(mask) >= 2:
                data[ei, ci] = np.interp(target, t[mask], y[mask], left=fill, right=fill)
    return EventAlignedTraces(
        time_relative_s=rel,
        data=data,
        channel_names=tuple(channel_names),
        event_times_s=event_times,
        event_metadata=meta,
    )


def event_aligned_summary_dataframe(aligned: EventAlignedTraces, *, statistics: Sequence[str] = ("mean", "std", "sem")):
    """Return one row per relative time and channel for aligned traces."""
    import pandas as pd
    from .statistics import nan_stats

    stats = nan_stats(aligned.data, axis=0)
    rows: list[dict[str, Any]] = []
    for ci, channel in enumerate(aligned.channel_names):
        for ti, trel in enumerate(aligned.time_relative_s):
            row: dict[str, Any] = {"channel": channel, "time_relative_s": float(trel), "n_events": int(stats["count"][ci, ti])}
            for stat in statistics:
                if stat == "count":
                    row["count"] = int(stats["count"][ci, ti])
                elif stat in stats:
                    row[stat] = float(stats[stat][ci, ti])
            rows.append(row)
    return pd.DataFrame(rows)


def event_aligned_traces_dataframe(aligned: EventAlignedTraces, *, max_events: int | None = None):
    """Return long-form event-aligned traces."""
    import pandas as pd

    n_events = aligned.n_events if max_events is None else min(aligned.n_events, int(max_events))
    rows: list[dict[str, Any]] = []
    for ei in range(n_events):
        meta = aligned.event_metadata[ei] if ei < len(aligned.event_metadata) else {}
        for ci, channel in enumerate(aligned.channel_names):
            vals = aligned.data[ei, ci]
            for ti, trel in enumerate(aligned.time_relative_s):
                v = vals[ti]
                rows.append({"event_i": ei, "event_time_s": float(aligned.event_times_s[ei]), "channel": channel, "time_relative_s": float(trel), "value": float(v) if np.isfinite(v) else np.nan, **meta})
    return pd.DataFrame(rows)


def select_events_by_spacing(
    event_times_s: Sequence[float],
    *,
    min_previous_interval_s: float | None = None,
    min_next_interval_s: float | None = None,
) -> np.ndarray:
    """Boolean mask selecting events with sufficient spacing."""
    t = np.asarray(event_times_s, dtype=float)
    keep = np.isfinite(t)
    if t.size == 0:
        return keep
    order = np.argsort(t)
    ts = t[order]
    keep_sorted = np.isfinite(ts)
    if min_previous_interval_s is not None:
        prev = np.diff(ts, prepend=-np.inf)
        keep_sorted &= prev >= float(min_previous_interval_s)
    if min_next_interval_s is not None:
        nxt = np.diff(ts, append=np.inf)
        keep_sorted &= nxt >= float(min_next_interval_s)
    keep[order] = keep_sorted
    return keep


def event_aligned_subset(aligned: EventAlignedTraces, mask: Sequence[bool]) -> EventAlignedTraces:
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] != aligned.n_events:
        raise ValueError("mask must have one value per event.")
    meta = tuple(x for x, ok in zip(aligned.event_metadata, m.tolist()) if ok)
    return EventAlignedTraces(aligned.time_relative_s, aligned.data[m], aligned.channel_names, aligned.event_times_s[m], meta)

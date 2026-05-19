from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..state import PhotometryState
from .events import (
    EventAlignedTraces,
    align_to_events,
    event_aligned_summary_dataframe,
    event_aligned_traces_dataframe,
    select_events_by_spacing,
)
from .peaks import PeakAnalysis, PeaksByTemplate, biexponential_peak
from .report import AnalysisResult, AnalysisWindow
from .statistics import nan_stats

DetectorKind = Literal["peaks", "template"]


def _normalise_channels(
    channels: Sequence[str] | str | None, state: PhotometryState
) -> tuple[str, ...]:
    if channels is None or channels == "all":
        return tuple(state.channel_names)
    if isinstance(channels, str):
        return tuple(
            c.strip().lower() for c in channels.split(",") if c.strip()
        )
    return tuple(str(c).lower() for c in channels)


def _window_dict(w: AnalysisWindow | None) -> dict[str, Any] | None:
    return None if w is None else w.as_dict()


def _accepted_kwargs(factory: Any) -> set[str]:
    try:
        return set(inspect.signature(factory).parameters)
    except (TypeError, ValueError):
        return set()


def _filter_detector_params(
    factory: Any, params: Mapping[str, Any]
) -> dict[str, Any]:
    """Drop stale detector parameters from a previous detector type.

    This keeps GUI workflows robust when switching between the template-matched
    detector and the local-peak detector. For example, ``template_params`` is
    valid for ``PeaksByTemplate`` but invalid for ``PeakAnalysis``. The GUI also
    updates these defaults, but this core guard prevents a stale config from
    crashing a batch analysis.
    """
    accepted = _accepted_kwargs(factory)
    if not accepted:
        return dict(params)
    return {k: v for k, v in dict(params).items() if k in accepted}


def _detector_from_config(
    *,
    detector: DetectorKind,
    event_signal: str,
    detector_params: Mapping[str, Any] | None,
    window: AnalysisWindow | None,
):
    params = dict(detector_params or {})
    params.setdefault("signal", event_signal)
    params.setdefault("window", window)
    if detector == "template":
        params.setdefault("func", biexponential_peak)
        return PeaksByTemplate(
            **_filter_detector_params(PeaksByTemplate, params)
        )
    if detector == "peaks":
        return PeakAnalysis(**_filter_detector_params(PeakAnalysis, params))
    raise ValueError("detector must be 'peaks' or 'template'.")


def result_to_event_aligned(result: AnalysisResult) -> EventAlignedTraces:
    arrays = result.arrays
    required = {
        "time_relative_s",
        "aligned_traces",
        "aligned_channel_names",
        "event_time_s",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(
            f"Result does not contain aligned traces. Missing: {sorted(missing)}"
        )
    return EventAlignedTraces(
        time_relative_s=np.asarray(arrays["time_relative_s"], dtype=float),
        data=np.asarray(arrays["aligned_traces"], dtype=float),
        channel_names=tuple(
            str(x) for x in np.asarray(arrays["aligned_channel_names"]).tolist()
        ),
        event_times_s=np.asarray(arrays["event_time_s"], dtype=float),
        event_metadata=tuple(
            {"event_i": int(i)}
            for i in range(len(np.asarray(arrays["event_time_s"])))
        ),
    )


def peak_triggered_summary_dataframe(result: AnalysisResult):
    aligned = result_to_event_aligned(result)
    df = event_aligned_summary_dataframe(
        aligned, statistics=("mean", "std", "sem", "median", "q25", "q75")
    )
    df.insert(0, "analysis", result.name)
    df.insert(1, "event_signal", result.channel)
    return df


def peak_triggered_traces_dataframe(
    result: AnalysisResult, *, max_events: int | None = None
):
    aligned = result_to_event_aligned(result)
    df = event_aligned_traces_dataframe(aligned, max_events=max_events)
    df.insert(0, "analysis", result.name)
    df.insert(1, "event_signal", result.channel)
    return df


@dataclass(frozen=True, slots=True)
class PeakTriggeredAverage:
    """Find peaks, align traces around them, and compute per-session averages."""

    event_signal: str
    channels: Sequence[str] | str | None = "all"
    detector: DetectorKind = "template"
    detector_params: Mapping[str, Any] | None = None
    window: AnalysisWindow | None = None
    t_before_s: float = 20.0
    t_after_s: float = 20.0
    target_fs: float | None = None
    dt: float | None = None
    exclude_times: Sequence[tuple[float, float]] | None = None
    exclude_mode: Literal["peak_time", "window_overlap"] = "peak_time"
    exclude_margin_s: float = 0.0
    min_previous_interval_s: float | None = None
    min_next_interval_s: float | None = None
    baseline_window_s: tuple[float, float] | None = None
    name: str = "peak_triggered_average"

    def __call__(self, state: PhotometryState) -> AnalysisResult:
        channels = _normalise_channels(self.channels, state)
        detector = _detector_from_config(
            detector=self.detector,
            event_signal=self.event_signal,
            detector_params=self.detector_params,
            window=self.window,
        )
        peak_res = detector(state)
        if not peak_res.arrays or "x" not in peak_res.arrays:
            return AnalysisResult(
                self.name,
                self.event_signal,
                self.window,
                self._params(),
                {"n_events": 0.0, "n_channels": float(len(channels))},
                {},
                "No peaks were detected for alignment.",
            )
        event_times = np.asarray(peak_res.arrays["x"], dtype=float)
        event_indices = np.arange(event_times.size, dtype=int)
        keep = np.isfinite(event_times)
        if self.exclude_times:
            windows = [
                (
                    float(a) - self.exclude_margin_s,
                    float(b) + self.exclude_margin_s,
                )
                for a, b in self.exclude_times
            ]
            if self.exclude_mode == "peak_time":
                for lo, hi in windows:
                    keep &= ~((event_times >= lo) & (event_times <= hi))
            elif self.exclude_mode == "window_overlap":
                for i, t0 in enumerate(event_times):
                    w0 = t0 - float(self.t_before_s)
                    w1 = t0 + float(self.t_after_s)
                    if any(a <= w1 and w0 <= b for a, b in windows):
                        keep[i] = False
            else:
                raise ValueError(
                    "exclude_mode must be 'peak_time' or 'window_overlap'."
                )
        if (
            self.min_previous_interval_s is not None
            or self.min_next_interval_s is not None
        ):
            keep &= select_events_by_spacing(
                event_times,
                min_previous_interval_s=self.min_previous_interval_s,
                min_next_interval_s=self.min_next_interval_s,
            )
        event_times_kept = event_times[keep]
        event_indices_kept = event_indices[keep]
        meta = tuple(
            {"source_peak_i": int(i)} for i in event_indices_kept.tolist()
        )
        aligned0 = align_to_events(
            state,
            event_times_kept,
            t_before=float(self.t_before_s),
            t_after=float(self.t_after_s),
            channels=channels,
            dt=self.dt,
            target_fs=self.target_fs,
        )
        aligned = EventAlignedTraces(
            aligned0.time_relative_s,
            aligned0.data,
            aligned0.channel_names,
            aligned0.event_times_s,
            meta,
        )
        data = aligned.data.copy()
        if self.baseline_window_s is not None and data.size:
            lo, hi = self.baseline_window_s
            m = (aligned.time_relative_s >= float(lo)) & (
                aligned.time_relative_s <= float(hi)
            )
            if np.any(m):
                data = data - np.nanmean(data[:, :, m], axis=2, keepdims=True)
                aligned = EventAlignedTraces(
                    aligned.time_relative_s,
                    data,
                    aligned.channel_names,
                    aligned.event_times_s,
                    aligned.event_metadata,
                )
        stats = nan_stats(data, axis=0)
        arrays: dict[str, Any] = {
            "time_relative_s": aligned.time_relative_s,
            "event_time_s": aligned.event_times_s,
            "source_peak_i": event_indices_kept,
            "aligned_channel_names": np.asarray(
                aligned.channel_names, dtype=str
            ),
            "aligned_traces": data,
            "detected_peak_time_s": event_times,
            "kept_peak_mask": keep,
        }
        for ci, channel in enumerate(aligned.channel_names):
            for stat_name in (
                "mean",
                "std",
                "sem",
                "median",
                "q25",
                "q75",
                "count",
            ):
                arrays[f"{stat_name}__{channel}"] = np.asarray(
                    stats[stat_name][ci]
                )
        metrics = {
            "n_events": float(aligned.n_events),
            "n_detected_events": float(event_times.size),
            "n_channels": float(len(aligned.channel_names)),
            "t_before_s": float(self.t_before_s),
            "t_after_s": float(self.t_after_s),
        }
        for ci, channel in enumerate(aligned.channel_names):
            vals = data[:, ci, :]
            metrics[f"{channel}_mean_peak_aligned_mean"] = (
                float(np.nanmean(vals)) if vals.size else np.nan
            )
            metrics[f"{channel}_mean_peak_aligned_max"] = (
                float(np.nanmax(stats["mean"][ci])) if vals.size else np.nan
            )
            metrics[f"{channel}_mean_peak_aligned_min"] = (
                float(np.nanmin(stats["mean"][ci])) if vals.size else np.nan
            )
        return AnalysisResult(
            self.name,
            self.event_signal,
            self.window,
            self._params(),
            metrics,
            arrays,
            "peak/event-triggered aligned traces and session average",
        )

    def _params(self) -> dict[str, Any]:
        return {
            "event_signal": self.event_signal,
            "channels": list(self.channels)
            if isinstance(self.channels, (list, tuple))
            else self.channels,
            "detector": self.detector,
            "detector_params": dict(self.detector_params or {}),
            "window": _window_dict(self.window),
            "t_before_s": self.t_before_s,
            "t_after_s": self.t_after_s,
            "target_fs": self.target_fs,
            "dt": self.dt,
            "exclude_times": list(self.exclude_times or []),
            "exclude_mode": self.exclude_mode,
            "exclude_margin_s": self.exclude_margin_s,
            "min_previous_interval_s": self.min_previous_interval_s,
            "min_next_interval_s": self.min_next_interval_s,
            "baseline_window_s": self.baseline_window_s,
        }

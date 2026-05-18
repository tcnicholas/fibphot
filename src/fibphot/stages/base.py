from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..state import HistoryPolicy, PhotometryState, StageRecord
from ..types import FloatArray


@dataclass(frozen=True, slots=True)
class StageOutput:
    """Output of a processing stage."""

    signals: FloatArray | None = None
    derived: dict[str, FloatArray] | None = None
    results: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    notes: str | None = None
    data: dict[str, object] = field(default_factory=dict)


def _resolve_channels(state: PhotometryState, channels: str | list[str] | None) -> list[int]:
    if channels is None or (isinstance(channels, str) and channels.lower() == "all"):
        return list(range(state.n_signals))
    if isinstance(channels, str):
        return [state.idx(channels)]
    return [state.idx(c) for c in channels]


@dataclass(frozen=True, slots=True)
class UpdateStage(ABC):
    name: str
    stage_plot: bool = False
    checkpoint: bool = False

    @abstractmethod
    def apply(self, state: PhotometryState) -> StageOutput:
        raise NotImplementedError

    def __call__(self, state: PhotometryState) -> PhotometryState:
        return self.run(state, history="all")

    def run(self, state: PhotometryState, *, history: HistoryPolicy = "all") -> PhotometryState:
        state0 = state.push_history(history, checkpoint=self.checkpoint)
        out = self.apply(state0)

        time_seconds = np.asarray(out.data.get("time_seconds", state0.time_seconds), dtype=float)
        history_arr = np.asarray(out.data.get("history", state0.history), dtype=float)
        new_signals = state0.signals if out.signals is None else np.asarray(out.signals, dtype=float)

        stage_id = f"{len(state0.summary) + 1:04d}_{self.name.lower()}"
        record = StageRecord(
            stage_id=stage_id,
            name=self.name,
            params=self._params_for_summary(),
            metrics=out.metrics or {},
            notes=out.notes,
        )

        new_derived = dict(state0.derived)
        if out.derived:
            new_derived.update(out.derived)

        new_results = dict(state0.results)
        new_results[stage_id] = out.results or {}

        return PhotometryState(
            time_seconds=time_seconds,
            signals=new_signals,
            channel_names=state0.channel_names,
            history=history_arr,
            summary=(*state0.summary, record),
            derived=new_derived,
            results=new_results,
            metadata=state0.metadata,
            readonly=state0.readonly,
        )

    def _params_for_summary(self) -> dict[str, Any]:
        return {}

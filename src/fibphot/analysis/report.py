from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy.typing as npt

from ..state import PhotometryState

WindowRef = Literal["seconds", "samples"]


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    """
    A window over which an analysis is evaluated.

    - ref="seconds": start/end are in seconds (state.time_seconds space)
    - ref="samples": start/end are integer sample indices [start, end)
    """
    start: float | int
    end: float | int
    ref: WindowRef = "seconds"
    label: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    name: str                       # e.g. "auc"
    channel: str                    # e.g. "gcamp"
    window: AnalysisWindow | None   # None means “whole trace”
    params: dict[str, Any] = field(default_factory=dict)

    metrics: dict[str, float] = field(default_factory=dict)
    arrays: dict[str, npt.NDArray[Any]] = field(default_factory=dict)

    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PhotometryReport:
    state: PhotometryState
    results: tuple[AnalysisResult, ...] = ()

    def add(self, result: AnalysisResult) -> PhotometryReport:
        return PhotometryReport(self.state, results=(*self.results, result))

    def extend(self, results: Iterable[AnalysisResult]) -> PhotometryReport:
        return PhotometryReport(self.state, results=(*self.results, *tuple(results)))

    def find(self, name: str) -> tuple[AnalysisResult, ...]:
        return tuple(r for r in self.results if r.name == name)
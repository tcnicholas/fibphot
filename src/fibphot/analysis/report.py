from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ..state import PhotometryState

WindowRef = Literal["seconds", "samples"]


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    start: float | int
    end: float | int
    ref: WindowRef = "seconds"
    label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "ref": self.ref, "label": self.label}


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    name: str
    channel: str
    window: AnalysisWindow | None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    arrays: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    notes: str | None = None

    @property
    def window_label(self) -> str:
        if self.window is None:
            return "whole_trace"
        if self.window.label:
            return self.window.label
        return f"{self.window.start}_{self.window.end}_{self.window.ref}"

    def metrics_row(self, state: PhotometryState | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "analysis": self.name,
            "channel": self.channel,
            "window": self.window_label,
        }
        if self.window is not None:
            row.update({
                "window_start": self.window.start,
                "window_end": self.window.end,
                "window_ref": self.window.ref,
            })
        if state is not None:
            row.update(_state_identity(state))
        row.update(self.metrics)
        return row

    def to_metrics_row(self, state: PhotometryState | None = None) -> dict[str, Any]:
        return self.metrics_row(state)

    def arrays_frame(self, state: PhotometryState | None = None):
        import pandas as pd
        if not self.arrays:
            return pd.DataFrame()
        arrays = {k: np.asarray(v) for k, v in self.arrays.items()}
        lengths = {k: int(v.shape[0]) for k, v in arrays.items() if v.ndim >= 1}
        if not lengths:
            row = {
                **(_state_identity(state) if state is not None else {}),
                "analysis": self.name,
                "channel": self.channel,
                "window": self.window_label,
            }
            for key, arr in arrays.items():
                if arr.ndim == 0:
                    row[key] = arr.item()
            return pd.DataFrame([row])
        n = max(lengths.values())
        rows: list[dict[str, Any]] = []
        ident = _state_identity(state) if state is not None else {}
        for i in range(n):
            row: dict[str, Any] = {
                **ident,
                "analysis": self.name,
                "channel": self.channel,
                "window": self.window_label,
                "row": i,
            }
            for key, arr in arrays.items():
                if arr.ndim == 0:
                    row[key] = arr.item()
                elif arr.ndim == 1:
                    if i < arr.shape[0]:
                        value = arr[i]
                        row[key] = value.item() if np.ndim(value) == 0 else value
                    else:
                        row[key] = np.nan
                elif arr.ndim == 2 and i < arr.shape[0]:
                    value = arr[i]
                    row[key] = value.item() if np.ndim(value) == 0 else value.tolist()
                # 3D+ arrays, such as event x channel x time trace stacks, are
                # intentionally exported through specialised long-form helpers.
            rows.append(row)
        return pd.DataFrame(rows)

    def to_arrays_frame(self, state: PhotometryState | None = None):
        return self.arrays_frame(state)


def _state_identity(state: PhotometryState) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subject": state.subject,
        "source_path": state.metadata.get("source_path"),
    }
    for k, v in state.tags.items():
        row[f"tag_{k}"] = v
    return row


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

    def metrics_dataframe(self):
        import pandas as pd
        return pd.DataFrame([r.metrics_row(self.state) for r in self.results])

    def arrays_dataframe(self):
        import pandas as pd
        frames = [r.arrays_frame(self.state) for r in self.results]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def to_dataframe(self):
        return self.metrics_dataframe()

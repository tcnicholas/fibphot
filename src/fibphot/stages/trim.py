from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..state import PhotometryState
from .base import StageOutput, UpdateStage

TrimUnit = Literal["samples", "seconds"]


def _as_n_samples(value: float, unit: TrimUnit, fs: float) -> int:
    if unit == "samples":
        return int(value)
    return int(round(float(value) * fs))


def _trim_derived_like_signals(
    derived: dict[str, Any],
    *,
    n_signals: int,
    n_samples: int,
    sl: slice,
) -> dict[str, Any]:
    """
    Trim derived arrays that match the signal shape conventions:
      - (n_signals, n_samples)
      - (n_samples,)
      - (h, n_signals, n_samples)  [rare, but we handle it]
    Leave everything else untouched.
    """
    out: dict[str, Any] = dict(derived)
    for k, v in derived.items():
        arr = (
            np.asarray(v) if isinstance(v, (list, tuple, np.ndarray)) else None
        )
        if arr is None:
            continue

        if arr.shape == (n_signals, n_samples):
            out[k] = arr[:, sl]
        elif arr.shape == (n_samples,):
            out[k] = arr[sl]
        elif arr.ndim == 3 and arr.shape[1:] == (n_signals, n_samples):
            out[k] = arr[:, :, sl]

    return out


@dataclass(frozen=True, slots=True)
class Trim(UpdateStage):
    """
    Discard datapoints from the start and/or end of the recording.

    You can specify trimming in either samples or seconds.

    Examples
    --------
    Trim 10 seconds from start:

        Trim(start=10, unit="seconds")

    Trim first 1 second and last 2 seconds:

        Trim(start=1, end=2, unit="seconds")

    Trim 100 samples from end:

        Trim(end=100, unit="samples")
    """

    name: str = field(default="trim", init=False)

    start: float = 0.0
    end: float = 0.0
    unit: TrimUnit = "seconds"

    def _params_for_summary(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "unit": self.unit}

    def apply(self, state: PhotometryState) -> StageOutput:
        if self.start < 0 or self.end < 0:
            raise ValueError("start/end must be >= 0.")

        fs = state.sampling_rate
        n0 = state.n_samples

        start_n = _as_n_samples(self.start, self.unit, fs)
        end_n = _as_n_samples(self.end, self.unit, fs)

        start_n = max(0, min(start_n, n0))
        end_n = max(0, min(end_n, n0))

        lo = start_n
        hi = n0 - end_n

        if hi <= lo:
            raise ValueError(
                "Trim removes all samples: "
                f"n={n0}, start={start_n}, end={end_n}."
            )

        sl = slice(lo, hi)

        new_time = state.time_seconds[sl]
        new_signals = state.signals[:, sl]
        new_history = (
            state.history[:, :, sl] if state.history.size else state.history
        )

        new_derived = _trim_derived_like_signals(
            state.derived,
            n_signals=state.n_signals,
            n_samples=state.n_samples,
            sl=sl,
        )

        return StageOutput(
            signals=new_signals,
            derived=new_derived,
            results={
                "unit": self.unit,
                "start": self.start,
                "end": self.end,
                "start_samples": int(start_n),
                "end_samples": int(end_n),
                "slice": (int(lo), int(hi)),
                "old_n_samples": int(n0),
                "new_n_samples": int(new_time.shape[0]),
            },
            notes=(
                "Trimmed time/signals "
                "(and any derived arrays matching signal shape)."
            ),
            data={
                "time_seconds": new_time,
                "history": new_history,
            },
        )

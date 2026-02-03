from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .types import FloatArray


@dataclass(frozen=True, slots=True)
class StageRecord:
    """Concise record of an applied preprocessing stage."""

    stage_id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class PhotometryState:
    """
    Immutable photometry data state.

    signals: stacked 2D array with shape (n_signals, n_samples).
    history: stacked 3D array with shape (h, n_signals, n_samples),
             storing previous *signals* snapshots only.
    """

    time_seconds: FloatArray
    signals: FloatArray
    channel_names: tuple[str, ...]
    history: FloatArray = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=float)
    )
    summary: tuple[StageRecord, ...] = ()
    derived: dict[str, FloatArray] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    _name_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t = np.asarray(self.time_seconds, dtype=float)
        s = np.asarray(self.signals, dtype=float)

        if t.ndim != 1:
            raise ValueError("time_seconds must be 1D.")
        if s.ndim != 2:
            raise ValueError(
                "signals must be 2D with shape (n_signals, n_samples)."
            )
        if s.shape[1] != t.shape[0]:
            raise ValueError(
                "signals second dimension must match time length: "
                f"{s.shape[1]} != {t.shape[0]}"
            )
        if len(self.channel_names) != s.shape[0]:
            raise ValueError(
                "channel_names length must match signals first dimension: "
                f"{len(self.channel_names)} != {s.shape[0]}"
            )

        # normalise channel names to lower case
        names = tuple(str(n).lower() for n in self.channel_names)
        name_to_idx = {n: i for i, n in enumerate(names)}

        # history normalisation
        h = np.asarray(self.history, dtype=float)
        if h.size == 0:
            h = np.empty((0, s.shape[0], s.shape[1]), dtype=float)
        elif h.ndim != 3 or h.shape[1:] != s.shape:
            raise ValueError(
                "history must be 3D with shape (h, n_signals, n_samples) "
                f"matching signals; got {h.shape}, expected "
                f"(*, {s.shape[0]}, {s.shape[1]})."
            )

        object.__setattr__(self, "time_seconds", t)
        object.__setattr__(self, "signals", s)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "history", h)
        object.__setattr__(self, "_name_to_index", name_to_idx)

    @property
    def n_samples(self) -> int:
        return int(self.time_seconds.shape[0])

    @property
    def n_signals(self) -> int:
        return int(self.signals.shape[0])

    @property
    def sampling_rate(self) -> float:
        dt = np.diff(self.time_seconds)
        return float(1.0 / np.median(dt))

    @property
    def tags(self) -> dict[str, str]:
        raw = self.metadata.get("tags", {})
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        return {}

    @property
    def subject(self) -> str | None:
        subj = self.metadata.get("subject")
        if subj is not None:
            return str(subj)

        src = self.metadata.get("source_path")
        if not src:
            return None

        stem = Path(str(src)).stem

        return stem.split("_", maxsplit=1)[0].lower() if stem else None

    def idx(self, channel: str) -> int:
        key = channel.lower()
        if key not in self._name_to_index:
            raise KeyError(
                f"Unknown channel '{channel}'. Available: {self.channel_names}"
            )
        return self._name_to_index[key]

    def channel(self, channel: str) -> FloatArray:
        return self.signals[self.idx(channel)]

    def tag(self, key: str, default: str | None = None) -> str | None:
        return self.tags.get(key, default)

    def with_channel(self, channel: str, values: FloatArray) -> PhotometryState:
        v = np.asarray(values, dtype=float)
        if v.shape != (self.n_samples,):
            raise ValueError(
                f"Channel replacement must have shape ({self.n_samples},), "
                f"got {v.shape}."
            )
        i = self.idx(channel)
        new_signals = self.signals.copy()
        new_signals[i] = v
        return replace(self, signals=new_signals)

    def with_metadata(
        self, updates: dict[str, Any] | None = None, **kwargs: Any
    ) -> PhotometryState:
        patch: dict[str, Any] = {}
        if updates:
            patch.update(updates)
        patch.update(kwargs)

        new_meta = dict(self.metadata)
        new_meta.update(patch)
        return replace(self, metadata=new_meta)

    def with_tags(
        self,
        tags: dict[str, str],
        *,
        overwrite: bool = False,
    ) -> PhotometryState:
        existing = dict(self.tags)
        incoming = {str(k): str(v) for k, v in tags.items()}

        if overwrite:
            existing.update(incoming)
        else:
            for k, v in incoming.items():
                existing.setdefault(k, v)

        return self.with_metadata(tags=existing)

    def push_history(self) -> PhotometryState:
        """Return a new state with the current signals appended to history."""
        new_hist = np.concatenate(
            [self.history, self.signals[None, :, :]], axis=0
        )
        return replace(self, history=new_hist)

    def raw(self) -> PhotometryState:
        """
        Return a new state representing the raw signals (after 0 stages).

        - signals restored to the raw snapshot
        - history cleared (you are back at the start)
        - summary/results/derived cleared
        - metadata preserved
        """
        if self.history.shape[0] == 0:
            # No stages applied, already raw.
            return self

        raw_signals = self.history[0]

        return PhotometryState(
            time_seconds=self.time_seconds,
            signals=raw_signals,
            channel_names=self.channel_names,
            history=np.empty((0, self.n_signals, self.n_samples), dtype=float),
            summary=(),
            derived={},
            results={},
            metadata=self.metadata,
        )

    def revert(self, n_steps: int | None = 1) -> PhotometryState:
        """
        Revert to a previous signals snapshot and drops corresponding summary
        entries and stage results.

        n_steps=1: before most recent stage
        n_steps=None: restore raw (after 0 stages)
        """
        if n_steps is None:
            return self.raw()

        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if self.history.shape[0] < n_steps:
            raise ValueError(
                f"Cannot revert {n_steps} step(s); history has "
                f"{self.history.shape[0]}."
            )

        restored_signals = self.history[-n_steps]
        new_history = self.history[:-n_steps]
        new_summary = self.summary[:-n_steps]

        valid_ids = {r.stage_id for r in new_summary}
        new_results = {k: v for k, v in self.results.items() if k in valid_ids}

        return PhotometryState(
            time_seconds=self.time_seconds,
            signals=restored_signals,
            channel_names=self.channel_names,
            history=new_history,
            summary=new_summary,
            derived={},
            results=new_results,
            metadata=self.metadata,
        )

    def revert_to(self, stage_name: str) -> PhotometryState:
        """
        Revert to the state immediately after the last occurrence of stage_name.

        If stage_name does not exist in the summary, raises KeyError.
        """
        target = stage_name.lower()
        names = [r.name.lower() for r in self.summary]
        if target not in names:
            raise KeyError(f"Stage '{stage_name}' not found in summary.")

        last_idx = max(i for i, n in enumerate(names) if n == target)
        steps_to_drop = len(self.summary) - (last_idx + 1)
        return self if steps_to_drop == 0 else self.revert(steps_to_drop)

    def pipe(self, *stages: Any) -> PhotometryState:
        """Apply stages in order (functional pipeline)."""
        state: PhotometryState = self
        for st in stages:
            state = st(state)
        return state

    def plot(
        self,
        *,
        signal: str,
        control: str | None = None,
        **kwargs: Any,
    ):
        """Plot the current state."""
        from .plotting import plot_current

        return plot_current(self, signal=signal, control=control, **kwargs)

    def plot_history(self, channel: str, **kwargs):
        """Plot a channel across the saved history (and optionally current)."""
        from .plotting import plot_history

        return plot_history(self, channel, **kwargs)

    def to_h5(
        self,
        path: Path | str,
        *,
        compression: str | None = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """Save this state to an HDF5 file."""
        from .io.h5 import save_state_h5

        save_state_h5(
            self,
            path,
            compression=compression,
            compression_opts=compression_opts,
        )

    @classmethod
    def from_h5(cls, path: Path | str) -> PhotometryState:
        """Load a state from an HDF5 file."""
        from .io.h5 import load_state_h5

        return load_state_h5(path)

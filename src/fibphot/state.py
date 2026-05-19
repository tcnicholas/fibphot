from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .misc import ReprText, sig, trunc_seq, uniform_repr
from .types import FloatArray

HistoryPolicy = Literal["none", "raw", "all", "checkpoints"]


def _compact_mapping(d: dict[str, Any], *, max_items: int = 10) -> ReprText:
    items = list(d.items())
    parts: list[str] = []
    for k, v in items[:max_items]:
        text = repr(v)
        if len(text) > 38:
            text = text[:35] + "..."
        parts.append(f"{k}={text}")
    if len(items) > max_items:
        parts.append(f"...+{len(items) - max_items}")
    return ReprText("{" + ", ".join(parts) + "}")


def _copy_array(a: FloatArray, *, deep: bool, readonly: bool) -> np.ndarray:
    out = np.array(a, dtype=float, copy=deep)
    if readonly:
        out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class StateValidation:
    ok: bool
    warnings: tuple[str, ...] = ()

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise ValueError(
                "Invalid PhotometryState: " + "; ".join(self.warnings)
            )


@dataclass(frozen=True, slots=True)
class StageRecord:
    """Concise record of an applied preprocessing stage."""

    stage_id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str | None = None

    def __repr__(self) -> str:
        info: dict[str, Any] = {"stage_id": self.stage_id, "name": self.name}
        if self.params:
            info["params"] = _compact_mapping(self.params, max_items=10)
        if self.metrics:
            info["metrics"] = _compact_mapping(self.metrics, max_items=10)
        if self.notes:
            note = self.notes.strip()
            info["notes"] = note if len(note) <= 80 else note[:77] + "..."
        return uniform_repr("StageRecord", **info, indent_width=4)


@dataclass(frozen=True, slots=True)
class PhotometryState:
    """
    Functional container for one fibre-photometry recording.

    The dataclass itself is frozen. Array mutability is explicit: pass
    readonly=True to prevent accidental in-place edits of arrays that are shared
    across states/history. For maximum speed, the default keeps arrays writable.
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
    readonly: bool = False

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

        names = tuple(str(n).strip().lower() for n in self.channel_names)
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Duplicate channel names after normalisation: {dupes}"
            )
        name_to_idx = {n: i for i, n in enumerate(names)}

        h = np.asarray(self.history, dtype=float)
        if h.size == 0:
            h = np.empty((0, s.shape[0], s.shape[1]), dtype=float)
        elif h.ndim != 3 or h.shape[1:] != s.shape:
            raise ValueError(
                "history must be 3D with shape (h, n_signals, n_samples) "
                f"matching signals; got {h.shape}, expected (*, {s.shape[0]}, {s.shape[1]})."
            )

        derived = {
            str(k): np.asarray(v, dtype=float) for k, v in self.derived.items()
        }

        if self.readonly:
            t = np.array(t, copy=True)
            s = np.array(s, copy=True)
            h = np.array(h, copy=True)
            t.setflags(write=False)
            s.setflags(write=False)
            h.setflags(write=False)
            for arr in derived.values():
                arr.setflags(write=False)

        object.__setattr__(self, "time_seconds", t)
        object.__setattr__(self, "signals", s)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "history", h)
        object.__setattr__(self, "derived", derived)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "results", dict(self.results))
        object.__setattr__(self, "_name_to_index", name_to_idx)

    @property
    def n_samples(self) -> int:
        return int(self.time_seconds.shape[0])

    @property
    def n_signals(self) -> int:
        return int(self.signals.shape[0])

    @property
    def sampling_rate(self) -> float:
        dt = np.diff(np.asarray(self.time_seconds, dtype=float))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            return float("nan")
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
        if subj is not None and str(subj):
            return str(subj)
        src = self.metadata.get("source_path") or self.metadata.get("file")
        if not src:
            return None
        stem = Path(str(src)).stem
        return stem.split("_", maxsplit=1)[0].lower() if stem else None

    def validate(
        self, *, regularity_rtol: float = 0.05, raise_on_error: bool = False
    ) -> StateValidation:
        warnings: list[str] = []
        t = np.asarray(self.time_seconds, dtype=float)
        s = np.asarray(self.signals, dtype=float)

        if t.ndim != 1:
            warnings.append("time_seconds is not 1D")
        if s.ndim != 2:
            warnings.append("signals is not 2D")
        if t.size != s.shape[1]:
            warnings.append(
                "time_seconds length does not match signals samples"
            )
        if not np.all(np.isfinite(t)):
            warnings.append("time_seconds contains non-finite values")
        if t.size >= 2:
            dt = np.diff(t)
            if np.any(dt <= 0):
                warnings.append("time_seconds is not strictly increasing")
            finite_dt = dt[np.isfinite(dt) & (dt > 0)]
            if finite_dt.size:
                med = float(np.median(finite_dt))
                max_rel = (
                    float(np.nanmax(np.abs(finite_dt - med) / med))
                    if med > 0
                    else float("inf")
                )
                if max_rel > regularity_rtol:
                    warnings.append(
                        f"sampling interval is irregular: max relative deviation {max_rel:.3g}"
                    )
        if s.size and not np.any(np.isfinite(s)):
            warnings.append("signals contain no finite values")
        if any(not n for n in self.channel_names):
            warnings.append("empty channel name present")
        if len(set(self.channel_names)) != len(self.channel_names):
            warnings.append("duplicate channel names present")

        out = StateValidation(ok=not warnings, warnings=tuple(warnings))
        if raise_on_error:
            out.raise_if_bad()
        return out

    def copy(
        self, *, deep: bool = True, readonly: bool | None = None
    ) -> PhotometryState:
        ro = self.readonly if readonly is None else bool(readonly)
        return PhotometryState(
            time_seconds=_copy_array(self.time_seconds, deep=deep, readonly=ro),
            signals=_copy_array(self.signals, deep=deep, readonly=ro),
            channel_names=self.channel_names,
            history=_copy_array(self.history, deep=deep, readonly=ro),
            summary=self.summary,
            derived={
                k: _copy_array(v, deep=deep, readonly=ro)
                for k, v in self.derived.items()
            },
            results=dict(self.results),
            metadata=dict(self.metadata),
            readonly=ro,
        )

    def as_readonly(self) -> PhotometryState:
        return self.copy(deep=True, readonly=True)

    def mutable_arrays(self) -> PhotometryState:
        return self.copy(deep=True, readonly=False)

    def idx(self, channel: str) -> int:
        key = channel.lower()
        if key not in self._name_to_index:
            raise KeyError(
                f"Unknown channel {channel!r}. Available: {self.channel_names}"
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
                f"Channel replacement must have shape ({self.n_samples},), got {v.shape}."
            )
        i = self.idx(channel)
        new_signals = np.array(self.signals, copy=True)
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
        self, tags: dict[str, str], *, overwrite: bool = False
    ) -> PhotometryState:
        existing = dict(self.tags)
        incoming = {str(k): str(v) for k, v in tags.items()}
        if overwrite:
            existing.update(incoming)
        else:
            for k, v in incoming.items():
                existing.setdefault(k, v)
        return self.with_metadata(tags=existing)

    def push_history(
        self, policy: HistoryPolicy = "all", *, checkpoint: bool = False
    ) -> PhotometryState:
        if policy == "none":
            return replace(
                self,
                history=np.empty(
                    (0, self.n_signals, self.n_samples), dtype=float
                ),
            )
        if policy == "raw":
            if self.history.shape[0] > 0:
                return self
            new_hist = self.signals[None, :, :]
        elif policy == "checkpoints":
            if not checkpoint:
                return self
            new_hist = np.concatenate(
                [self.history, self.signals[None, :, :]], axis=0
            )
        elif policy == "all":
            new_hist = np.concatenate(
                [self.history, self.signals[None, :, :]], axis=0
            )
        else:
            raise ValueError(f"Unknown history policy: {policy!r}")
        return replace(self, history=np.asarray(new_hist, dtype=float))

    def raw(self) -> PhotometryState:
        if self.history.shape[0] == 0:
            return self
        return PhotometryState(
            time_seconds=self.time_seconds,
            signals=self.history[0],
            channel_names=self.channel_names,
            history=np.empty((0, self.n_signals, self.n_samples), dtype=float),
            summary=(),
            derived={},
            results={},
            metadata=self.metadata,
            readonly=self.readonly,
        )

    def revert(self, n_steps: int | None = 1) -> PhotometryState:
        if n_steps is None:
            return self.raw()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if self.history.shape[0] < n_steps:
            raise ValueError(
                f"Cannot revert {n_steps} step(s); history has {self.history.shape[0]}."
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
            readonly=self.readonly,
        )

    def revert_to(self, stage_name: str) -> PhotometryState:
        target = stage_name.lower()
        names = [r.name.lower() for r in self.summary]
        if target not in names:
            raise KeyError(f"Stage {stage_name!r} not found in summary.")
        last_idx = max(i for i, n in enumerate(names) if n == target)
        steps_to_drop = len(self.summary) - (last_idx + 1)
        return self if steps_to_drop == 0 else self.revert(steps_to_drop)

    def pipe(
        self, *stages: Any, history: HistoryPolicy = "all"
    ) -> PhotometryState:
        state: PhotometryState = self
        for st in stages:
            run = getattr(st, "run", None)
            state = run(state, history=history) if callable(run) else st(state)
        return state

    def pipe_with_plots(
        self,
        *stages: Any,
        channels: list[str] | None = None,
        control: str | None = None,
        title: str | None = None,
        history: HistoryPolicy = "all",
    ) -> tuple[PhotometryState, object | None]:
        state: PhotometryState = self
        requested_ids: list[str] = []
        for st in stages:
            run = getattr(st, "run", None)
            state = run(state, history=history) if callable(run) else st(state)
            if bool(getattr(st, "stage_plot", False)) and state.summary:
                requested_ids.append(state.summary[-1].stage_id)
        if not requested_ids:
            return state, None
        from .plotting import plot_pipeline_overview

        fig, _ = plot_pipeline_overview(
            state,
            stage_ids=requested_ids,
            channels=channels,
            control=control,
            title=title,
        )
        return state, fig

    def plot(self, *, signal: str, control: str | None = None, **kwargs: Any):
        from .plotting import plot_current

        return plot_current(self, signal=signal, control=control, **kwargs)

    def plot_history(self, channel: str, **kwargs: Any):
        from .plotting import plot_history

        return plot_history(self, channel, **kwargs)

    def analyse(self, *analyses: Any):
        from .analysis.report import PhotometryReport

        report = PhotometryReport(self)
        for analysis in analyses:
            out = analysis(self)
            if isinstance(out, (list, tuple)):
                report = report.extend(out)
            else:
                report = report.add(out)
        return report

    def to_h5(
        self,
        path: Path | str,
        *,
        compression: str | None = "gzip",
        compression_opts: int = 4,
    ) -> None:
        from .io.h5 import save_state_h5

        save_state_h5(
            self,
            path,
            compression=compression,
            compression_opts=compression_opts,
        )

    @classmethod
    def from_h5(cls, path: Path | str) -> PhotometryState:
        from .io.h5 import load_state_h5

        return load_state_h5(path)

    def __repr__(self) -> str:
        info: dict[str, Any] = {}
        if self.subject:
            info["subject"] = self.subject
        src = self.metadata.get("source_path") or self.metadata.get("file")
        if src:
            info["source"] = Path(str(src)).name
        info["n_signals"] = self.n_signals
        info["n_samples"] = self.n_samples
        if self.n_samples >= 2:
            duration = float(self.time_seconds[-1] - self.time_seconds[0])
            info["duration_s"] = sig(duration, 4)
            fs = self.sampling_rate
            if np.isfinite(fs):
                info["fs_hz"] = sig(fs, 4)
        info["channels"] = trunc_seq(self.channel_names, max_items=10)
        info["history"] = int(self.history.shape[0])
        if self.readonly:
            info["readonly"] = True
        if self.summary:
            info["stages"] = len(self.summary)
            info["last_stage"] = self.summary[-1].name
            names = [r.name for r in self.summary]
            info["pipeline"] = (
                tuple(names)
                if len(names) <= 5
                else (names[0], names[1], "...", names[-1])
            )
        if self.derived:
            info["derived"] = trunc_seq(tuple(self.derived), max_items=10)
        if self.results:
            info["results"] = trunc_seq(tuple(self.results), max_items=10)
        if self.tags:
            kv = [f"{k}={v}" for k, v in list(self.tags.items())[:8]]
            more = len(self.tags) - len(kv)
            if more > 0:
                kv.append(f"...+{more}")
            info["tags"] = ReprText("{" + ", ".join(kv) + "}")
        return uniform_repr(
            "PhotometryState", **info, indent_width=4, max_width=88
        )

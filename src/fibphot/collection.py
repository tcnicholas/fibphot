from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .analysis.aggregate import (
    AlignedSignals,
    TraceStatistics,
    align_collection_signals,
    mean_state_from_aligned,
    trace_statistics_from_aligned,
)
from .analysis.report import AnalysisResult, PhotometryReport
from .misc import ReprText, sig, trunc_seq, uniform_repr
from .state import HistoryPolicy, PhotometryState
from .tags import TagTable, apply_tags, default_subject_getter, read_tag_table

SubjectGetter = Callable[[PhotometryState], str | None]
AlignMode = Literal["intersection", "union"]
InterpKind = Literal["linear", "nearest"]
TimeRef = Literal["absolute", "start"]


def metadata_value(state: PhotometryState, key: str, default: Any = "") -> Any:
    """Look up a grouping/filtering value from metadata, tags or properties."""
    if key in {"subject", "mouse", "mouse_id"}:
        return state.subject or state.metadata.get("subject", default)
    if key in state.tags:
        return state.tags.get(key, default)
    if key in state.metadata:
        return state.metadata.get(key, default)
    if key.startswith("tag_"):
        return state.tags.get(key[4:], default)
    if key == "source_parent":
        sp = state.metadata.get("source_path")
        return Path(sp).parent.name if sp else default
    if key.startswith("source_parent_"):
        sp = state.metadata.get("source_path")
        if not sp:
            return default
        try:
            depth = int(key.rsplit("_", 1)[1])
        except Exception:
            return default
        parents = Path(sp).parents
        return parents[depth].name if depth < len(parents) else default
    return default


def metadata_keys(states: Sequence[PhotometryState]) -> tuple[str, ...]:
    keys: set[str] = {"subject", "source_path", "source_name", "source_parent"}
    for s in states:
        keys.update(str(k) for k in s.metadata)
        keys.update(f"tag_{k}" for k in s.tags)
        keys.update(str(k) for k in s.tags)
    return tuple(sorted(keys))


def _source_metadata(
    path: Path, *, base_path: Path | None = None
) -> dict[str, Any]:
    p = path.resolve()
    meta: dict[str, Any] = {
        "source_path": str(p),
        "source_name": p.name,
        "source_stem": p.stem,
        "source_parent": p.parent.name,
        "source_parent_path": str(p.parent),
        "subject": p.stem.split("_", maxsplit=1)[0].lower(),
    }
    for i, parent in enumerate(p.parents[:6]):
        meta[f"source_parent_{i}"] = parent.name
    if base_path is not None:
        try:
            rel = p.relative_to(base_path.resolve())
            meta["source_relative_path"] = str(rel)
            for i, part in enumerate(rel.parts[:-1]):
                meta[f"source_relative_parent_{i}"] = part
        except Exception:
            pass
    return meta


@dataclass(frozen=True, slots=True)
class BatchFailure:
    index: int
    subject: str | None
    source_path: str | None
    step: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class BatchResult:
    successful: PhotometryCollection
    failed: tuple[BatchFailure, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failed

    def failures_table(self):
        import pandas as pd

        return pd.DataFrame([f.__dict__ for f in self.failed])


@dataclass(frozen=True, slots=True)
class BatchReport:
    reports: tuple[PhotometryReport, ...]
    failed: tuple[BatchFailure, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failed

    def metrics_dataframe(self):
        import pandas as pd

        frames = [r.metrics_dataframe() for r in self.reports]
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    def arrays_dataframe(self):
        import pandas as pd

        frames = [r.arrays_dataframe() for r in self.reports]
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    def failures_table(self):
        import pandas as pd

        return pd.DataFrame([f.__dict__ for f in self.failed])

    def grouped_metrics(
        self,
        groupby: str | Sequence[str] | None = None,
        *,
        values: Sequence[str] | None = None,
    ):
        from .analysis.statistics import grouped_numeric_summary

        return grouped_numeric_summary(
            self.metrics_dataframe(), groupby=groupby, values=values
        )

    def grouped_arrays(
        self,
        groupby: str | Sequence[str] | None = None,
        *,
        values: Sequence[str] | None = None,
    ):
        from .analysis.statistics import grouped_numeric_summary

        return grouped_numeric_summary(
            self.arrays_dataframe(), groupby=groupby, values=values
        )


@dataclass(frozen=True, slots=True)
class PhotometryCollection:
    """Immutable wrapper around multiple PhotometryState objects."""

    states: Sequence[PhotometryState]

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))

    @classmethod
    def from_iterable(
        cls, states: Iterable[PhotometryState]
    ) -> PhotometryCollection:
        return cls(states=tuple(states))

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[Path | str],
        *,
        reader: Callable[[Path], PhotometryState],
        metadata_fn: Callable[[Path], dict[str, Any]] | None = None,
        base_path: Path | str | None = None,
        sort: bool = True,
    ) -> PhotometryCollection:
        path_list = [Path(p) for p in paths]
        if sort:
            path_list.sort()
        base = Path(base_path) if base_path is not None else None
        states: list[PhotometryState] = []
        for p in path_list:
            st = reader(p)
            meta = _source_metadata(p, base_path=base)
            if st.subject:
                meta["subject"] = st.subject
            if metadata_fn is not None:
                meta.update(metadata_fn(p) or {})
            states.append(st.with_metadata(meta))
        return cls(states=tuple(states))

    @classmethod
    def from_glob(
        cls,
        base_path: Path | str,
        pattern: str = "*.doric",
        *,
        reader: Callable[[Path], PhotometryState],
        metadata_fn: Callable[[Path], dict[str, Any]] | None = None,
        sort: bool = True,
        recursive: bool = False,
    ) -> PhotometryCollection:
        base = Path(base_path)
        paths = list(base.rglob(pattern) if recursive else base.glob(pattern))
        return cls.from_paths(
            paths,
            reader=reader,
            metadata_fn=metadata_fn,
            base_path=base,
            sort=sort,
        )

    @classmethod
    def from_directory(
        cls,
        base_path: Path | str,
        *,
        pattern: str = "**/*.doric",
        reader: Callable[[Path], PhotometryState],
        metadata_fn: Callable[[Path], dict[str, Any]] | None = None,
        sort: bool = True,
    ) -> PhotometryCollection:
        base = Path(base_path)
        return cls.from_paths(
            base.glob(pattern),
            reader=reader,
            metadata_fn=metadata_fn,
            base_path=base,
            sort=sort,
        )

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self):
        return iter(self.states)

    def __getitem__(
        self, item: int | slice
    ) -> PhotometryState | PhotometryCollection:
        if isinstance(item, slice):
            return PhotometryCollection(self.states[item])
        return self.states[item]

    @property
    def subjects(self) -> tuple[str | None, ...]:
        return tuple(s.subject for s in self.states)

    def pipe(
        self, *stages: object, history: HistoryPolicy = "raw"
    ) -> PhotometryCollection:
        return PhotometryCollection(
            states=tuple(s.pipe(*stages, history=history) for s in self.states)
        )

    def batch_pipe(
        self,
        *stages: object,
        history: HistoryPolicy = "raw",
        continue_on_error: bool = True,
    ) -> BatchResult:
        ok: list[PhotometryState] = []
        failures: list[BatchFailure] = []
        for i, state in enumerate(self.states):
            current = state
            try:
                for st in stages:
                    step = getattr(st, "name", st.__class__.__name__)
                    run = getattr(st, "run", None)
                    current = (
                        run(current, history=history)
                        if callable(run)
                        else st(current)
                    )
                ok.append(current)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    BatchFailure(
                        index=i,
                        subject=state.subject,
                        source_path=str(state.metadata.get("source_path"))
                        if state.metadata.get("source_path")
                        else None,
                        step=step if "step" in locals() else "unknown",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                if not continue_on_error:
                    raise
        return BatchResult(
            successful=PhotometryCollection(ok), failed=tuple(failures)
        )

    def analyse(
        self, *analyses: object, continue_on_error: bool = True
    ) -> BatchReport:
        reports: list[PhotometryReport] = []
        failures: list[BatchFailure] = []
        for i, state in enumerate(self.states):
            try:
                reports.append(state.analyse(*analyses))
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    BatchFailure(
                        index=i,
                        subject=state.subject,
                        source_path=str(state.metadata.get("source_path"))
                        if state.metadata.get("source_path")
                        else None,
                        step="analysis",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                if not continue_on_error:
                    raise
        return BatchReport(reports=tuple(reports), failed=tuple(failures))

    def with_tags_from_table(
        self,
        path: Path | str,
        *,
        subject_getter: SubjectGetter = default_subject_getter,
        overwrite: bool = False,
    ) -> PhotometryCollection:
        return self.with_tags(
            read_tag_table(path),
            subject_getter=subject_getter,
            overwrite=overwrite,
        )

    def with_tags(
        self,
        table: TagTable,
        *,
        subject_getter: SubjectGetter = default_subject_getter,
        overwrite: bool = False,
    ) -> PhotometryCollection:
        tagged = tuple(
            apply_tags(
                s, table, subject_getter=subject_getter, overwrite=overwrite
            )
            for s in self.states
        )
        return PhotometryCollection(states=tagged)

    def with_metadata_table(
        self,
        path: Path | str,
        *,
        on: str = "subject",
        overwrite: bool = True,
        as_tags: bool = True,
    ) -> PhotometryCollection:
        """Merge a CSV/Excel metadata table into each state."""
        import pandas as pd

        p = Path(path)
        table = (
            pd.read_excel(p)
            if p.suffix.lower() in {".xlsx", ".xls"}
            else pd.read_csv(p)
        )
        if on not in table.columns:
            raise ValueError(
                f"Metadata table does not contain match column {on!r}."
            )
        lookup = {str(row[on]): row.to_dict() for _, row in table.iterrows()}
        out: list[PhotometryState] = []
        for state in self.states:
            key = str(metadata_value(state, on, ""))
            extra = lookup.get(key, {})
            if not extra:
                out.append(state)
                continue
            meta = dict(state.metadata)
            tags = dict(state.tags)
            for k, v in extra.items():
                if k == on:
                    continue
                vv = None if pd.isna(v) else v
                if overwrite or k not in meta:
                    meta[k] = vv
                if as_tags and (overwrite or k not in tags):
                    tags[k] = "" if vv is None else str(vv)
            meta["tags"] = tags
            out.append(state.with_metadata(meta))
        return PhotometryCollection(states=tuple(out))

    @property
    def metadata_keys(self) -> tuple[str, ...]:
        return metadata_keys(self.states)

    def filter(
        self,
        predicate: Callable[[PhotometryState], bool] | None = None,
        **criteria: Any,
    ) -> PhotometryCollection:
        """Filter by metadata or tags."""

        def match_value(value: Any, criterion: Any) -> bool:
            if callable(criterion):
                return bool(criterion(value))
            if isinstance(criterion, (set, list, tuple)) and not isinstance(
                criterion, str
            ):
                return value in criterion
            return value == criterion

        def ok(s: PhotometryState) -> bool:
            if predicate is not None and not predicate(s):
                return False
            return all(
                match_value(metadata_value(s, k, None), v)
                for k, v in criteria.items()
            )

        return PhotometryCollection(
            states=tuple(s for s in self.states if ok(s))
        )

    def groupby(
        self, key: str | Sequence[str] | Callable[[PhotometryState], str]
    ) -> dict[str, PhotometryCollection]:
        groups: dict[str, list[PhotometryState]] = {}
        for s in self.states:
            if callable(key):
                v = str(key(s))
            elif isinstance(key, str):
                v = str(metadata_value(s, key, ""))
            else:
                v = "|".join(str(metadata_value(s, k, "")) for k in key)
            groups.setdefault(v, []).append(s)
        return {
            k: PhotometryCollection(states=tuple(v)) for k, v in groups.items()
        }

    def sort_by(self, *keys: str) -> PhotometryCollection:
        def sort_key(s: PhotometryState) -> tuple[str, ...]:
            return tuple(str(metadata_value(s, k, "")) for k in keys)

        return PhotometryCollection(
            states=tuple(sorted(self.states, key=sort_key))
        )

    def map(
        self, fn: Callable[[PhotometryState], PhotometryState]
    ) -> PhotometryCollection:
        return PhotometryCollection(states=tuple(fn(s) for s in self.states))

    def summary_table(self):
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for i, s in enumerate(self.states):
            row: dict[str, Any] = {
                "index": i,
                "subject": s.subject,
                "source_path": s.metadata.get("source_path"),
                "n_samples": s.n_samples,
                "n_signals": s.n_signals,
                "sampling_rate_hz": s.sampling_rate,
                "duration_s": float(s.time_seconds[-1] - s.time_seconds[0])
                if s.n_samples >= 2
                else np.nan,
                "channels": ",".join(s.channel_names),
                "n_stages": len(s.summary),
            }
            for k, v in s.metadata.items():
                if k not in row and isinstance(
                    v, (str, int, float, bool, type(None))
                ):
                    row[k] = v
            row.update({f"tag_{k}": v for k, v in s.tags.items()})
            for k, v in s.tags.items():
                row.setdefault(k, v)
            rows.append(row)
        return pd.DataFrame(rows)

    def align(
        self,
        *,
        channels: Sequence[str] | None = None,
        align: AlignMode = "intersection",
        time_ref: TimeRef = "start",
        dt: float | None = None,
        target_fs: float | None = None,
        interpolation: InterpKind = "linear",
        fill: float = float("nan"),
    ) -> AlignedSignals:
        return align_collection_signals(
            self.states,
            channels=channels,
            align=align,
            time_ref=time_ref,
            dt=dt,
            target_fs=target_fs,
            interpolation=interpolation,
            fill=fill,
        )

    def mean(
        self,
        *,
        channels: Sequence[str] | None = None,
        align: AlignMode = "intersection",
        time_ref: TimeRef = "start",
        dt: float | None = None,
        target_fs: float | None = None,
        interpolation: InterpKind = "linear",
        fill: float = float("nan"),
        name: str = "group_mean",
    ) -> PhotometryState:
        return mean_state_from_aligned(
            self.align(
                channels=channels,
                align=align,
                time_ref=time_ref,
                dt=dt,
                target_fs=target_fs,
                interpolation=interpolation,
                fill=fill,
            ),
            name=name,
        )

    def trace_statistics(
        self,
        *,
        channels: Sequence[str] | None = None,
        align: AlignMode = "intersection",
        time_ref: TimeRef = "start",
        dt: float | None = None,
        target_fs: float | None = None,
        interpolation: InterpKind = "linear",
        fill: float = float("nan"),
    ) -> TraceStatistics:
        """Return aligned individual traces and mean/std/SEM statistics.

        This is the core backend used by the GUI batch trace overlay/average
        view.  The full aligned stack is retained so callers can display raw
        individual traces, the average, or both with uncertainty bands.
        """

        return trace_statistics_from_aligned(
            self.align(
                channels=channels,
                align=align,
                time_ref=time_ref,
                dt=dt,
                target_fs=target_fs,
                interpolation=interpolation,
                fill=fill,
            )
        )

    def grouped_trace_statistics(
        self,
        by: str | Sequence[str],
        *,
        channels: Sequence[str] | None = None,
        align: AlignMode = "intersection",
        time_ref: TimeRef = "start",
        dt: float | None = None,
        target_fs: float | None = None,
        interpolation: InterpKind = "linear",
        fill: float = float("nan"),
    ) -> dict[str, TraceStatistics]:
        """Return trace statistics for every metadata group."""

        return {
            key: group.trace_statistics(
                channels=channels,
                align=align,
                time_ref=time_ref,
                dt=dt,
                target_fs=target_fs,
                interpolation=interpolation,
                fill=fill,
            )
            for key, group in self.groupby(by).items()
        }

    def grouped_mean(
        self,
        by: str | Sequence[str],
        *,
        channels: Sequence[str] | None = None,
        align: AlignMode = "intersection",
        time_ref: TimeRef = "start",
        dt: float | None = None,
        target_fs: float | None = None,
        interpolation: InterpKind = "linear",
        fill: float = float("nan"),
    ) -> dict[str, PhotometryState]:
        """Average traces separately for every metadata group."""
        return {
            key: group.mean(
                channels=channels,
                align=align,
                time_ref=time_ref,
                dt=dt,
                target_fs=target_fs,
                interpolation=interpolation,
                fill=fill,
                name=f"group_mean:{key}",
            )
            for key, group in self.groupby(by).items()
        }

    def to_h5(
        self,
        path: Path | str,
        *,
        compression: str | None = "gzip",
        compression_opts: int = 4,
    ) -> None:
        from .io.h5 import save_collection_h5

        save_collection_h5(
            self,
            path,
            compression=compression,
            compression_opts=compression_opts,
        )

    @classmethod
    def from_h5(cls, path: Path | str) -> PhotometryCollection:
        from .io.h5 import load_collection_h5

        return load_collection_h5(path)

    def __repr__(self) -> str:
        info: dict[str, Any] = {"n_states": len(self.states)}
        if not self.states:
            return uniform_repr("PhotometryCollection", **info, indent_width=4)
        subjects = [s.subject for s in self.states]
        uniq = sorted(set(x for x in subjects if x is not None))
        info["n_subjects"] = len(uniq)
        if uniq:
            info["subjects"] = trunc_seq(uniq, max_items=8)
        sets = [set(s.channel_names) for s in self.states]
        inter = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        info["channels_intersection"] = tuple(sorted(inter)) if inter else ()
        info["channels_union_n"] = len(union)
        depths = np.array([len(s.summary) for s in self.states], dtype=int)
        info["stages_med"] = int(np.median(depths))
        info["stages_minmax"] = ReprText(
            f"({int(depths.min())}, {int(depths.max())})"
        )
        fs = np.array([s.sampling_rate for s in self.states], dtype=float)
        fs = fs[np.isfinite(fs)]
        if fs.size:
            info["fs_hz_med"] = sig(float(np.median(fs)), 4)
            info["fs_hz_minmax"] = ReprText(
                f"({sig(float(fs.min()), 4)}, {sig(float(fs.max()), 4)})"
            )
        dur = np.array(
            [
                float(s.time_seconds[-1] - s.time_seconds[0])
                if s.n_samples >= 2
                else 0.0
                for s in self.states
            ],
            dtype=float,
        )
        info["duration_s_med"] = sig(float(np.median(dur)), 4)
        info["duration_s_minmax"] = ReprText(
            f"({sig(float(dur.min()), 4)}, {sig(float(dur.max()), 4)})"
        )
        tag_keys: dict[str, int] = {}
        for s in self.states:
            for k in s.tags:
                tag_keys[k] = tag_keys.get(k, 0) + 1
        if tag_keys:
            items = sorted(tag_keys.items(), key=lambda kv: (-kv[1], kv[0]))
            parts = [f"{k}:{c}/{len(self.states)}" for k, c in items[:6]]
            if len(items) > 6:
                parts.append(f"...+{len(items) - 6}")
            info["tags"] = ReprText("{" + ", ".join(parts) + "}")
        return uniform_repr(
            "PhotometryCollection", **info, indent_width=4, max_width=88
        )

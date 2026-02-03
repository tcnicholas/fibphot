from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .analysis.aggregate import (
    AlignedSignals,
    align_collection_signals,
    mean_state_from_aligned,
)
from .misc import (
    ReprText,
    sig,
    trunc_seq,
    uniform_repr,
)
from .state import PhotometryState
from .tags import TagTable, apply_tags, default_subject_getter, read_tag_table

SubjectGetter = Callable[[PhotometryState], str | None]
AlignMode = Literal["intersection", "union"]
InterpKind = Literal["linear", "nearest"]
TimeRef = Literal["absolute", "start"]


@dataclass(frozen=True, slots=True)
class PhotometryCollection:
    """
    A thin immutable wrapper around multiple PhotometryState objects.

    Provides tag-aware filtering, grouping, sorting, and bulk serialisation.
    """

    states: Sequence[PhotometryState]

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))

    @classmethod
    def from_iterable(
        cls, states: Iterable[PhotometryState]
    ) -> PhotometryCollection:
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
    ) -> PhotometryCollection:
        base = Path(base_path)
        paths = list(base.glob(pattern))
        if sort:
            paths.sort()

        def _iter() -> Iterable[PhotometryState]:
            for p in paths:
                st = reader(p)
                if metadata_fn is not None:
                    st = st.with_metadata(**metadata_fn(p))
                yield st

        return cls(states=tuple(_iter()))

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self):
        return iter(self.states)

    @property
    def subjects(self) -> tuple[str | None, ...]:
        return tuple(s.subject for s in self.states)

    def pipe(self, *stages: object) -> PhotometryCollection:
        return PhotometryCollection(
            states=tuple(s.pipe(*stages) for s in self.states)
        )

    def with_tags_from_table(
        self,
        path: Path | str,
        *,
        subject_getter: SubjectGetter = default_subject_getter,
        overwrite: bool = False,
    ) -> PhotometryCollection:
        table = read_tag_table(path)
        return self.with_tags(
            table, subject_getter=subject_getter, overwrite=overwrite
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

    def filter(self, **criteria: str) -> PhotometryCollection:
        """
        Filter by tags: e.g. .filter(genotype="KO", context="A")
        """

        def ok(s: PhotometryState) -> bool:
            tags = s.tags
            return all(tags.get(k) == v for k, v in criteria.items())

        return PhotometryCollection(
            states=tuple(s for s in self.states if ok(s))
        )

    def groupby(self, key: str) -> dict[str, PhotometryCollection]:
        """
        Group by a tag key: returns {tag_value: PhotometryCollection}.
        Missing values go under "".
        """
        groups: dict[str, list[PhotometryState]] = {}
        for s in self.states:
            v = s.tags.get(key, "")
            groups.setdefault(v, []).append(s)

        return {
            k: PhotometryCollection(states=tuple(v)) for k, v in groups.items()
        }

    def sort_by(self, *keys: str) -> PhotometryCollection:
        """
        Sort by one or more tag keys (lexicographic).
        Missing values sort as "".
        """

        def sort_key(s: PhotometryState) -> tuple[str, ...]:
            tags = s.tags
            return tuple(tags.get(k, "") for k in keys)

        return PhotometryCollection(
            states=tuple(sorted(self.states, key=sort_key))
        )

    def map(
        self, fn: Callable[[PhotometryState], PhotometryState]
    ) -> PhotometryCollection:
        return PhotometryCollection(states=tuple(fn(s) for s in self.states))

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
        aligned = self.align(
            channels=channels,
            align=align,
            time_ref=time_ref,
            dt=dt,
            target_fs=target_fs,
            interpolation=interpolation,
            fill=fill,
        )
        return mean_state_from_aligned(aligned, name=name)

    def to_h5(self, path: Path | str) -> None:
        from .io.h5 import save_collection_h5

        save_collection_h5(self, path)

    @classmethod
    def from_h5(cls, path: Path | str) -> PhotometryCollection:
        from .io.h5 import load_collection_h5

        return load_collection_h5(path)

    def __repr__(self) -> str:
        info: dict[str, Any] = {}

        n = len(self.states)
        info["n_states"] = n
        if n == 0:
            return uniform_repr("PhotometryCollection", **info, indent_width=4)

        # ---- subjects summary
        subjects = [s.subject for s in self.states]
        known = [x for x in subjects if x is not None]
        uniq = sorted(set(known))
        info["n_subjects"] = len(uniq)

        if uniq:
            shown = trunc_seq(uniq, max_items=8)
            info["subjects"] = shown

        # ---- channels summary (intersection/union)
        sets = [set(s.channel_names) for s in self.states]
        inter = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        info["channels_intersection"] = tuple(sorted(inter)) if inter else ()
        info["channels_union_n"] = len(union)

        # ---- stage depth summary
        depths = np.array([len(s.summary) for s in self.states], dtype=int)
        info["stages_med"] = int(np.median(depths))
        info["stages_minmax"] = ReprText(
            f"({int(depths.min())}, {int(depths.max())})"
        )

        # ---- sampling rate / duration summary
        fs = np.array([s.sampling_rate for s in self.states], dtype=float)
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

        # ---- tag-key coverage summary
        tag_keys: dict[str, int] = {}
        for s in self.states:
            for k in s.tags:
                tag_keys[k] = tag_keys.get(k, 0) + 1

        if tag_keys:
            items = sorted(tag_keys.items(), key=lambda kv: (-kv[1], kv[0]))
            shown = items[:6]
            parts = [f"{k}:{c}/{n}" for k, c in shown]
            if len(items) > len(shown):
                parts.append(f"...+{len(items) - len(shown)}")
            info["tags"] = ReprText("{" + ", ".join(parts) + "}")

        return uniform_repr(
            "PhotometryCollection", **info, indent_width=4, max_width=88
        )

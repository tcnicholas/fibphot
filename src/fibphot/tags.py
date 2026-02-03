from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .state import PhotometryState

SubjectGetter = Callable[[PhotometryState], str | None]


@dataclass(frozen=True, slots=True)
class TagTable:
    """
    Mapping from subject -> tags.

    The first column is assumed to be the subject identifier. Remaining columns
    are treated as tag keys.
    """

    by_subject: dict[str, dict[str, str]]

    def tags_for(self, subject: str) -> dict[str, str]:
        return dict(self.by_subject.get(subject.lower(), {}))


def read_tag_table(path: Path | str) -> TagTable:
    """
    Read a delimited text table where the first column is subject.

    Supports CSV/TSV/space-delimited. Header row is required.
    """
    path = Path(path)

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Tag file is empty: {path}")

    # sniff delimiter (fallback to whitespace splitting)
    sample = "\n".join(text.splitlines()[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        delim: str | None = dialect.delimiter
    except csv.Error:
        delim = None

    rows: list[list[str]] = []
    if delim is None:
        for line in text.splitlines():
            rows.append(re.split(r"\s+", line.strip()))
    else:
        reader = csv.reader(text.splitlines(), delimiter=delim)
        rows = [list(r) for r in reader]

    if len(rows) < 2:
        raise ValueError(
            "Tag table must include a header and at least one row."
        )

    header = [h.strip().lower() for h in rows[0]]
    if len(header) < 2:
        raise ValueError("Tag table must have >=2 columns (subject + tags).")

    subj_key = header[0]
    tag_keys = header[1:]

    mapping: dict[str, dict[str, str]] = {}
    for r in rows[1:]:
        if not r or all(not c.strip() for c in r):
            continue
        subject = str(r[0]).strip().lower()
        if not subject:
            continue
        tags: dict[str, str] = {}
        for j, k in enumerate(tag_keys, start=1):
            if j >= len(r):
                tags[k] = ""
            else:
                tags[k] = str(r[j]).strip()
        mapping[subject] = tags

    if subj_key != "subject":
        pass

    return TagTable(by_subject=mapping)


def subject_from_filename(path: Path | str) -> str:
    """
    Default subject extraction:
      - take the stem
      - split on '_' and use the first token

    Example:
      '74R_10122025_A61603_0001.doric' -> '74r'
    """
    p = Path(path)
    stem = p.stem
    token = stem.split("_", maxsplit=1)[0]
    return token.lower()


def default_subject_getter(state: PhotometryState) -> str | None:
    """
    Prefer explicit metadata subject, else derive from metadata['source_path'].
    """
    if state.subject is not None:
        return state.subject
    src = state.metadata.get("source_path")
    if not src:
        return None
    return subject_from_filename(str(src))


def apply_tags(
    state: PhotometryState,
    table: TagTable,
    *,
    subject_getter: SubjectGetter = default_subject_getter,
    overwrite: bool = False,
) -> PhotometryState:
    subject = subject_getter(state)
    if subject is None:
        return state
    tags = table.tags_for(subject)
    if not tags:
        return state
    return state.with_tags(tags, overwrite=overwrite)

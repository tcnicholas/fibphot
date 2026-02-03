from __future__ import annotations

from pathlib import Path


def subject_from_stem(p: Path) -> str:
    return p.stem.split("_", 1)[0]

def metadata_from_stem(p: Path) -> dict[str, object]:
    return {"subject": subject_from_stem(p)}
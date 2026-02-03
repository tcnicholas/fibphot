from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping


def subject_from_stem(p: Path) -> str:
    return p.stem.split("_", 1)[0]


def metadata_from_stem(p: Path) -> dict[str, object]:
    return {"subject": subject_from_stem(p)}


def uniform_repr(
    thing_name: str,
    *anonymous_things: Any,
    max_width: int = 88,
    stringify: bool = True,
    indent_width: int = 4,
    **named_things: Any,
) -> str:
    def _to_str(thing: Any) -> str:
        if isinstance(thing, str) and stringify:
            return f'"{thing}"'
        return str(thing)

    info = list(map(_to_str, anonymous_things))
    info += [f"{name}={_to_str(thing)}" for name, thing in named_things.items()]

    single_liner = f"{thing_name}({', '.join(info)})"
    if len(single_liner) < max_width and "\n" not in single_liner:
        return single_liner

    def indent(s: str) -> str:
        _indent = " " * indent_width
        return "\n".join(f"{_indent}{line}" for line in s.split("\n"))

    rep = f"{thing_name}("
    for thing in info:
        rep += "\n" + indent(thing) + ","

    return rep[:-1] + "\n)"


@dataclass(frozen=True, slots=True)
class ReprText:
    """Small wrapper so uniform_repr won't auto-quote preformatted strings."""

    text: str

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


def sig(x: float, n: int = 3) -> ReprText:
    """Significant-figures formatting with sane handling of NaN/inf."""
    try:
        xf = float(x)
    except Exception:
        return ReprText(str(x))
    if not isfinite(xf):
        return ReprText(str(xf))
    return ReprText(format(xf, f".{n}g"))


def trunc_seq(seq: Iterable[Any], max_items: int = 8) -> tuple[Any, ...]:
    items = list(seq)
    if len(items) <= max_items:
        return tuple(items)
    keep = max_items - 1
    return tuple(items[:keep]) + (f"...+{len(items) - keep}",)


def trunc_keys(d: Mapping[str, Any], max_items: int = 8) -> tuple[str, ...]:
    return trunc_seq(list(d.keys()), max_items=max_items)

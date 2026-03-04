from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..collection import PhotometryCollection
from ..state import PhotometryState
from ..types import FloatArray
from .report import AnalysisResult

TimeWindow = tuple[float, float]
ExcludeMode = Literal["peak_time", "window_overlap"]


def _as_float_1d(x: FloatArray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return a


def _normalise_windows(
    windows: Sequence[TimeWindow] | None,
) -> list[TimeWindow]:
    if not windows:
        return []
    out: list[TimeWindow] = []
    for a, b in windows:
        a_f = float(a)
        b_f = float(b)
        if not np.isfinite(a_f) or not np.isfinite(b_f):
            continue
        lo, hi = (a_f, b_f) if a_f <= b_f else (b_f, a_f)
        out.append((lo, hi))
    out.sort(key=lambda x: x[0])

    merged: list[TimeWindow] = []
    for lo, hi in out:
        if not merged or lo > merged[-1][1]:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    return merged


def _expand_windows(
    windows: Sequence[TimeWindow], margin_s: float
) -> list[TimeWindow]:
    m = float(margin_s)
    if m <= 0:
        return list(windows)
    return _normalise_windows([(lo - m, hi + m) for lo, hi in windows])


def _in_any_window(t: float, windows: Sequence[TimeWindow]) -> bool:
    return any(lo <= t <= hi for lo, hi in windows)


def _overlaps_any_window(w: TimeWindow, windows: Sequence[TimeWindow]) -> bool:
    w0, w1 = w
    lo, hi = (w0, w1) if w0 <= w1 else (w1, w0)
    return any(a <= hi and lo <= b for a, b in windows)


def _trim_derived_like_signals(
    derived: dict[str, Any],
    *,
    n_signals: int,
    n_samples: int,
    sl: slice,
) -> dict[str, Any]:
    """Trim derived arrays that match signal-like shapes.

    Shapes handled:
      - (n_signals, n_samples)
      - (n_samples,)
      - (h, n_signals, n_samples)
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


def _slice_by_seconds(
    state: PhotometryState,
    *,
    start_s: float,
    end_s: float,
    pad: Literal["error", "clip"] = "clip",
) -> tuple[slice, np.ndarray]:
    t = _as_float_1d(state.time_seconds)
    lo_s = float(min(start_s, end_s))
    hi_s = float(max(start_s, end_s))

    if not np.any(np.isfinite(t)):
        raise ValueError("time_seconds contains no finite values.")

    i0 = int(np.searchsorted(t, lo_s, side="left"))
    i1 = int(np.searchsorted(t, hi_s, side="right"))

    if pad == "clip":
        i0 = max(0, min(i0, state.n_samples))
        i1 = max(0, min(i1, state.n_samples))

    if i1 <= i0:
        raise ValueError(
            f"Requested window [{lo_s}, {hi_s}] contains no samples."
        )

    return slice(i0, i1), t[i0:i1]


@dataclass(frozen=True, slots=True)
class PeakEpoch:
    """One extracted peak-centred epoch."""

    peak_i: int
    peak_time_s: float
    window_s: TimeWindow
    state: PhotometryState


def extract_peak_epochs(
    state: PhotometryState,
    res: AnalysisResult,
    *,
    t_before_s: float = 5.0,
    t_after_s: float = 5.0,
    shift_time: bool = True,
    # exclusion
    exclude_times: Sequence[TimeWindow] | None = None,
    exclude_margin_s: float = 0.0,
    exclude_mode: ExcludeMode = "peak_time",
    # selection
    keep_only: Iterable[int] | None = None,
    kind: Literal["peak", "valley", "both"] = "both",
    # output
    as_collection: bool = True,
    keep_history: bool = False,
    keep_summary: bool = True,
    keep_results: bool = False,
    keep_derived: bool = False,
    pad: Literal["error", "clip"] = "clip",
) -> PhotometryCollection | tuple[PeakEpoch, ...]:
    """Extract fixed windows around peaks found by PeakAnalysis.

    Uses res.arrays['x'] for peak times (seconds) and res.arrays['kind'].
    """

    if res.name != "peaks":
        raise ValueError(
            "extract_peak_epochs expects an AnalysisResult from PeakAnalysis "
            "(res.name == 'peaks')."
        )

    if not res.arrays or "x" not in res.arrays:
        out = PhotometryCollection(states=())
        return out if as_collection else ()

    t_before = float(t_before_s)
    t_after = float(t_after_s)
    if t_before < 0 or t_after < 0:
        raise ValueError("t_before_s and t_after_s must be non-negative")

    x_all = _as_float_1d(res.arrays["x"])

    if "kind" in res.arrays:
        kinds_all = np.asarray(res.arrays["kind"], dtype=str)
        if kinds_all.shape != x_all.shape:
            raise ValueError(
                "res.arrays['kind'] must match res.arrays['x'] shape"
            )
    else:
        kinds_all = np.full(x_all.shape, "peak", dtype=str)

    if kind != "both":
        m_kind = kinds_all == kind
        x = x_all[m_kind]
        keep_idx_map = np.flatnonzero(m_kind)
    else:
        x = x_all
        keep_idx_map = np.arange(x_all.shape[0], dtype=int)

    if keep_only is not None:
        keep_set = {int(i) for i in keep_only}
        m_keep = np.array(
            [int(i) in keep_set for i in keep_idx_map], dtype=bool
        )
        x = x[m_keep]
        keep_idx_map = keep_idx_map[m_keep]

    excl = _expand_windows(_normalise_windows(exclude_times), exclude_margin_s)

    epochs: list[PeakEpoch] = []
    for global_i, t_pk in zip(keep_idx_map.tolist(), x.tolist()):
        t_pk_f = float(t_pk)
        if not np.isfinite(t_pk_f):
            continue

        w = (t_pk_f - t_before, t_pk_f + t_after)

        if excl:
            if exclude_mode == "peak_time":
                if _in_any_window(t_pk_f, excl):
                    continue
            elif exclude_mode == "window_overlap":
                if _overlaps_any_window(w, excl):
                    continue
            else:
                raise ValueError(f"Unknown exclude_mode: {exclude_mode!r}")

        sl, new_time = _slice_by_seconds(
            state, start_s=w[0], end_s=w[1], pad=pad
        )

        new_signals = state.signals[:, sl]

        if keep_history and state.history.size:
            new_history = state.history[:, :, sl]
        else:
            new_history = np.empty(
                (0, state.n_signals, new_time.shape[0]), dtype=float
            )

        new_summary = state.summary if keep_summary else ()
        new_results = state.results if keep_results else {}
        new_derived = (
            _trim_derived_like_signals(
                state.derived,
                n_signals=state.n_signals,
                n_samples=state.n_samples,
                sl=sl,
            )
            if keep_derived
            else {}
        )

        t_out = new_time
        if shift_time:
            t_out = t_out - t_pk_f

        meta = dict(state.metadata)
        meta.update(
            {
                "epoch": {
                    "source": "peaks",
                    "peak_index": int(global_i),
                    "peak_time_s": float(t_pk_f),
                    "window_s": (float(w[0]), float(w[1])),
                    "shift_time": bool(shift_time),
                }
            }
        )

        ep_state = PhotometryState(
            time_seconds=t_out,
            signals=new_signals,
            channel_names=state.channel_names,
            history=new_history,
            summary=new_summary,
            derived=new_derived,
            results=new_results,
            metadata=meta,
        )

        epochs.append(
            PeakEpoch(
                peak_i=int(global_i),
                peak_time_s=float(t_pk_f),
                window_s=(float(w[0]), float(w[1])),
                state=ep_state,
            )
        )

    if as_collection:
        return PhotometryCollection(states=tuple(e.state for e in epochs))

    return tuple(epochs)

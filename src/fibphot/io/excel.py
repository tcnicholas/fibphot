from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..state import PhotometryState
from ..types import FloatArray


def _stack_signals(signals: Mapping[str, FloatArray]) -> tuple[FloatArray, tuple[str, ...]]:
    names = tuple(str(k).lower() for k in signals)
    arrs = [np.asarray(signals[n], dtype=float) for n in signals]
    stacked = np.stack(arrs, axis=0)
    return stacked, names


def read_excel(
    filename: Path | str,
    *,
    time_column: str = "time",
    signal_columns: Sequence[str] | Mapping[str, str] | None = ("gcamp", "isosbestic"),
) -> PhotometryState:
    """
    Read photometry data from an Excel file into a PhotometryState.

    signal_columns:
      - Sequence[str]: load these columns as channels (channel name == column name)
      - Mapping[str, str]: {channel_name: column_name_in_excel}
      - None: load all columns except the time column
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path)
    df.columns = [str(c).lower() for c in df.columns]

    tcol = time_column.lower()
    if tcol not in df.columns:
        raise ValueError(f"Missing time column '{time_column}'. Found: {df.columns}")

    if signal_columns is None:
        column_map = {c: c for c in df.columns if c != tcol}
    elif isinstance(signal_columns, Mapping):
        column_map = {str(k).lower(): str(v).lower() for k, v in signal_columns.items()}
    else:
        cols = [str(c).lower() for c in signal_columns]
        column_map = {c: c for c in cols}

    missing = [col for col in column_map.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing signal columns: {missing}. Found: {df.columns}")

    signals_dict = {ch: df[col].to_numpy(dtype=float) for ch, col in column_map.items()}
    signals, names = _stack_signals(signals_dict)
    time_s = df[tcol].to_numpy(dtype=float)

    return PhotometryState(time_seconds=time_s, signals=signals, channel_names=names)

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

StatName = Literal["count", "mean", "std", "sem", "median", "min", "max", "q25", "q75"]


def nan_count(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.sum(np.isfinite(x), axis=axis)


def nan_sem(x: np.ndarray, axis: int | None = None, ddof: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = nan_count(x, axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=ddof)
    return sd / np.sqrt(np.maximum(n, 1))


def nan_stats(x: np.ndarray, *, axis: int | None = 0, ddof: int = 1) -> dict[str, np.ndarray]:
    arr = np.asarray(x, dtype=float)
    return {
        "count": np.asarray(nan_count(arr, axis=axis)),
        "mean": np.nanmean(arr, axis=axis),
        "std": np.nanstd(arr, axis=axis, ddof=ddof),
        "sem": nan_sem(arr, axis=axis, ddof=ddof),
        "median": np.nanmedian(arr, axis=axis),
        "q25": np.nanpercentile(arr, 25, axis=axis),
        "q75": np.nanpercentile(arr, 75, axis=axis),
        "min": np.nanmin(arr, axis=axis),
        "max": np.nanmax(arr, axis=axis),
    }


def grouped_numeric_summary(
    frame: Any,
    *,
    groupby: str | Sequence[str] | None = None,
    values: Sequence[str] | None = None,
    statistics: Sequence[StatName] = ("count", "mean", "std", "sem"),
):
    import pandas as pd

    df = pd.DataFrame(frame).copy()
    if df.empty:
        return df
    if groupby is None or groupby == "":
        group_cols: list[str] = []
    elif isinstance(groupby, str):
        group_cols = [c.strip() for c in groupby.split(",") if c.strip()]
    else:
        group_cols = list(groupby)
    group_cols = [c for c in group_cols if c in df.columns]

    if values is None:
        excluded = {"index", "row", "event_i", "sample", "n_samples", "n_signals"}
        values = [c for c in df.select_dtypes(include="number").columns if c not in excluded and not c.endswith("_index")]
    value_cols = [c for c in values if c in df.columns]
    if not value_cols:
        return pd.DataFrame()

    iterator = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    rows: list[dict[str, Any]] = []
    for key, g in iterator:
        if not isinstance(key, tuple):
            key = (key,)
        base = {col: val for col, val in zip(group_cols, key)}
        for value_col in value_cols:
            vals = pd.to_numeric(g[value_col], errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            row: dict[str, Any] = {**base, "metric": value_col}
            if "count" in statistics:
                row["count"] = int(finite.size)
            if finite.size:
                if "mean" in statistics:
                    row["mean"] = float(np.mean(finite))
                if "std" in statistics:
                    row["std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                if "sem" in statistics:
                    row["sem"] = float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0
                if "median" in statistics:
                    row["median"] = float(np.median(finite))
                if "min" in statistics:
                    row["min"] = float(np.min(finite))
                if "max" in statistics:
                    row["max"] = float(np.max(finite))
                if "q25" in statistics:
                    row["q25"] = float(np.percentile(finite, 25))
                if "q75" in statistics:
                    row["q75"] = float(np.percentile(finite, 75))
            else:
                for stat in statistics:
                    if stat != "count":
                        row[stat] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)

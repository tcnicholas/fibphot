from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..collection import PhotometryCollection
from ..state import PhotometryState, StageRecord


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": obj.shape,
            "data": obj.tolist(),
        }
    return str(obj)


def _json_dumps(obj: Any) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False)


def _json_loads(s: str) -> Any:
    return json.loads(s)


def _require_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise ImportError("Saving/loading HDF5 requires `h5py`.") from exc
    return h5py


def save_state_h5(
    state: PhotometryState,
    path: Path | str,
    *,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    h5py = _require_h5py()
    path = Path(path)

    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "fibphot_state"
        f.attrs["schema_version"] = 1
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        f.create_dataset("time_seconds", data=state.time_seconds)
        f.create_dataset(
            "signals",
            data=state.signals,
            compression=compression,
            compression_opts=compression_opts,
        )
        f.create_dataset(
            "history",
            data=state.history,
            compression=compression,
            compression_opts=compression_opts,
        )

        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "channel_names",
            data=np.array(state.channel_names, dtype=object),
            dtype=dt,
        )

        # metadata (json)
        f.attrs["metadata_json"] = _json_dumps(state.metadata)

        # derived arrays
        g_derived = f.create_group("derived")
        for k, arr in state.derived.items():
            g_derived.create_dataset(
                k,
                data=np.asarray(arr, dtype=float),
                compression=compression,
                compression_opts=compression_opts,
            )

        # summary records
        g_sum = f.create_group("summary")
        for rec in state.summary:
            g = g_sum.create_group(rec.stage_id)
            g.attrs["name"] = rec.name
            g.attrs["params_json"] = _json_dumps(rec.params)
            g.attrs["metrics_json"] = _json_dumps(rec.metrics)
            if rec.notes is not None:
                g.attrs["notes"] = rec.notes

        # results by stage_id
        g_res = f.create_group("results")
        for stage_id, payload in state.results.items():
            g = g_res.create_group(stage_id)
            g.attrs["json"] = _json_dumps(payload)


def load_state_h5(path: Path | str) -> PhotometryState:
    h5py = _require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as f:
        t = np.asarray(f["time_seconds"], dtype=float)
        s = np.asarray(f["signals"], dtype=float)
        h = np.asarray(f["history"], dtype=float)
        channel_names = tuple(str(x) for x in f["channel_names"][...])

        metadata = {}
        meta_json = f.attrs.get("metadata_json")
        if meta_json:
            metadata = _json_loads(str(meta_json))

        derived: dict[str, np.ndarray] = {}
        if "derived" in f:
            for k in f["derived"].keys():
                derived[k] = np.asarray(f["derived"][k], dtype=float)

        summary: list[StageRecord] = []
        if "summary" in f:
            for stage_id in f["summary"].keys():
                g = f["summary"][stage_id]
                name = str(g.attrs["name"])
                params = _json_loads(str(g.attrs.get("params_json", "{}")))
                metrics = _json_loads(str(g.attrs.get("metrics_json", "{}")))
                notes = g.attrs.get("notes")
                summary.append(
                    StageRecord(
                        stage_id=str(stage_id),
                        name=name,
                        params=params if isinstance(params, dict) else {},
                        metrics=metrics if isinstance(metrics, dict) else {},
                        notes=str(notes) if notes is not None else None,
                    )
                )
            summary.sort(key=lambda r: r.stage_id)

        results: dict[str, dict[str, Any]] = {}
        if "results" in f:
            for stage_id in f["results"].keys():
                g = f["results"][stage_id]
                payload = _json_loads(str(g.attrs.get("json", "{}")))
                results[str(stage_id)] = (
                    payload if isinstance(payload, dict) else {}
                )

        return PhotometryState(
            time_seconds=t,
            signals=s,
            channel_names=channel_names,
            history=h,
            summary=tuple(summary),
            derived=derived,
            results=results,
            metadata=metadata,
        )


def save_collection_h5(
    coll: PhotometryCollection,
    path: Path | str,
    *,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    h5py = _require_h5py()
    path = Path(path)

    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "fibphot_collection"
        f.attrs["schema_version"] = 1
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        g_states = f.create_group("states")
        order: list[str] = []

        for i, st in enumerate(coll.states):
            name = st.subject or f"state_{i:04d}"
            base = name
            j = 1
            while name in g_states:
                j += 1
                name = f"{base}_{j}"
            order.append(name)

            g = g_states.create_group(name)
            g.create_dataset("time_seconds", data=st.time_seconds)
            g.create_dataset(
                "signals",
                data=st.signals,
                compression=compression,
                compression_opts=compression_opts,
            )
            g.create_dataset(
                "history",
                data=st.history,
                compression=compression,
                compression_opts=compression_opts,
            )

            dt = h5py.string_dtype(encoding="utf-8")
            g.create_dataset(
                "channel_names",
                data=np.array(st.channel_names, dtype=object),
                dtype=dt,
            )
            g.attrs["metadata_json"] = _json_dumps(st.metadata)

            gd = g.create_group("derived")
            for k, arr in st.derived.items():
                gd.create_dataset(
                    k,
                    data=np.asarray(arr, dtype=float),
                    compression=compression,
                    compression_opts=compression_opts,
                )

            gs = g.create_group("summary")
            for rec in st.summary:
                gg = gs.create_group(rec.stage_id)
                gg.attrs["name"] = rec.name
                gg.attrs["params_json"] = _json_dumps(rec.params)
                gg.attrs["metrics_json"] = _json_dumps(rec.metrics)
                if rec.notes is not None:
                    gg.attrs["notes"] = rec.notes

            gr = g.create_group("results")
            for stage_id, payload in st.results.items():
                gg = gr.create_group(stage_id)
                gg.attrs["json"] = _json_dumps(payload)

        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("order", data=np.array(order, dtype=object), dtype=dt)


def load_collection_h5(path: Path | str) -> PhotometryCollection:
    h5py = _require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as f:
        order = [str(x) for x in f["order"][...]]
        g_states = f["states"]

        states: list[PhotometryState] = []
        for key in order:
            g = g_states[key]
            t = np.asarray(g["time_seconds"], dtype=float)
            s = np.asarray(g["signals"], dtype=float)
            h = np.asarray(g["history"], dtype=float)
            channel_names = tuple(str(x) for x in g["channel_names"][...])

            metadata = {}
            meta_json = g.attrs.get("metadata_json")
            if meta_json:
                metadata = _json_loads(str(meta_json))

            derived: dict[str, np.ndarray] = {}
            if "derived" in g:
                for k in g["derived"].keys():
                    derived[k] = np.asarray(g["derived"][k], dtype=float)

            summary: list[StageRecord] = []
            if "summary" in g:
                for stage_id in g["summary"].keys():
                    gg = g["summary"][stage_id]
                    name = str(gg.attrs["name"])
                    params = _json_loads(str(gg.attrs.get("params_json", "{}")))
                    metrics = _json_loads(
                        str(gg.attrs.get("metrics_json", "{}"))
                    )
                    notes = gg.attrs.get("notes")
                    summary.append(
                        StageRecord(
                            stage_id=str(stage_id),
                            name=name,
                            params=params if isinstance(params, dict) else {},
                            metrics=metrics
                            if isinstance(metrics, dict)
                            else {},
                            notes=str(notes) if notes is not None else None,
                        )
                    )
                summary.sort(key=lambda r: r.stage_id)

            results: dict[str, dict[str, Any]] = {}
            if "results" in g:
                for stage_id in g["results"].keys():
                    gg = g["results"][stage_id]
                    payload = _json_loads(str(gg.attrs.get("json", "{}")))
                    results[str(stage_id)] = (
                        payload if isinstance(payload, dict) else {}
                    )

            states.append(
                PhotometryState(
                    time_seconds=t,
                    signals=s,
                    channel_names=channel_names,
                    history=h,
                    summary=tuple(summary),
                    derived=derived,
                    results=results,
                    metadata=metadata,
                )
            )

        return PhotometryCollection.from_iterable(states)

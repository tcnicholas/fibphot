from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _compression_kwargs(compression: str | None, compression_opts: int) -> dict[str, Any]:
    if compression is None:
        return {}
    return {"compression": compression, "compression_opts": compression_opts}

import numpy as np

from ..state import PhotometryState, StageRecord


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, np.ndarray):
        return {"__array_ref__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    return str(obj)


def _json_dumps(obj: Any) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False)


def _json_loads(s: str | bytes) -> Any:
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return json.loads(str(s))


def _require_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise ImportError("Saving/loading HDF5 requires `h5py`.") from exc
    return h5py


def _decode_str(x: Any) -> str:
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def _split_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    meta: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray):
            arrays[str(k)] = np.asarray(v)
        elif isinstance(v, dict):
            sub_meta, sub_arrays = _split_payload(v)
            meta[str(k)] = sub_meta
            for kk, arr in sub_arrays.items():
                arrays[f"{k}/{kk}"] = arr
        else:
            meta[str(k)] = v
    return meta, arrays


def _write_payload_group(g: Any, payload: dict[str, Any], *, compression: str | None, compression_opts: int) -> None:
    meta, arrays = _split_payload(payload)
    g.attrs["attrs_json"] = _json_dumps(meta)
    ga = g.create_group("arrays")
    for key, arr in arrays.items():
        parts = key.split("/")
        gg = ga
        for part in parts[:-1]:
            gg = gg.require_group(part)
        gg.create_dataset(parts[-1], data=np.asarray(arr), **_compression_kwargs(compression, compression_opts))


def _read_arrays_group(g: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    def visit(name: str, obj: Any) -> None:
        if hasattr(obj, "shape"):
            out[name] = np.asarray(obj)
    g.visititems(visit)
    return out


def _read_payload_group(g: Any) -> dict[str, Any]:
    payload = _json_loads(g.attrs.get("attrs_json", "{}"))
    if not isinstance(payload, dict):
        payload = {}
    if "arrays" in g:
        for key, arr in _read_arrays_group(g["arrays"]).items():
            parts = key.split("/")
            dest = payload
            for part in parts[:-1]:
                dest = dest.setdefault(part, {})
            dest[parts[-1]] = arr
    return payload


def _write_state_group(g: Any, state: PhotometryState, *, compression: str | None, compression_opts: int) -> None:
    h5py = _require_h5py()
    g.create_dataset("time_seconds", data=state.time_seconds)
    g.create_dataset("signals", data=state.signals, **_compression_kwargs(compression, compression_opts))
    g.create_dataset("history", data=state.history, **_compression_kwargs(compression, compression_opts))
    dt = h5py.string_dtype(encoding="utf-8")
    g.create_dataset("channel_names", data=np.array(state.channel_names, dtype=object), dtype=dt)
    g.attrs["metadata_json"] = _json_dumps(state.metadata)
    g.attrs["readonly"] = bool(state.readonly)

    gd = g.create_group("derived")
    for k, arr in state.derived.items():
        gd.create_dataset(k, data=np.asarray(arr), **_compression_kwargs(compression, compression_opts))

    gs = g.create_group("summary")
    order: list[str] = []
    for rec in state.summary:
        order.append(rec.stage_id)
        gg = gs.create_group(rec.stage_id)
        gg.attrs["name"] = rec.name
        gg.attrs["params_json"] = _json_dumps(rec.params)
        gg.attrs["metrics_json"] = _json_dumps(rec.metrics)
        if rec.notes is not None:
            gg.attrs["notes"] = rec.notes
    g.create_dataset("summary_order", data=np.array(order, dtype=object), dtype=dt)

    gr = g.create_group("results")
    for result_id, payload in state.results.items():
        _write_payload_group(gr.create_group(str(result_id)), payload, compression=compression, compression_opts=compression_opts)


def _read_state_group(g: Any) -> PhotometryState:
    t = np.asarray(g["time_seconds"], dtype=float)
    s = np.asarray(g["signals"], dtype=float)
    h = np.asarray(g["history"], dtype=float)
    channel_names = tuple(_decode_str(x) for x in g["channel_names"][...])
    metadata: dict[str, Any] = {}
    meta_json = g.attrs.get("metadata_json")
    if meta_json:
        tmp = _json_loads(meta_json)
        metadata = tmp if isinstance(tmp, dict) else {}
    derived: dict[str, np.ndarray] = {}
    if "derived" in g:
        for k in g["derived"].keys():
            derived[str(k)] = np.asarray(g["derived"][k], dtype=float)
    summary: list[StageRecord] = []
    order = [_decode_str(x) for x in g["summary_order"][...]] if "summary_order" in g else list(g.get("summary", {}).keys())
    if "summary" in g:
        for stage_id in order:
            if stage_id not in g["summary"]:
                continue
            gg = g["summary"][stage_id]
            params = _json_loads(gg.attrs.get("params_json", "{}"))
            metrics = _json_loads(gg.attrs.get("metrics_json", "{}"))
            notes = gg.attrs.get("notes")
            summary.append(StageRecord(
                stage_id=str(stage_id),
                name=_decode_str(gg.attrs["name"]),
                params=params if isinstance(params, dict) else {},
                metrics=metrics if isinstance(metrics, dict) else {},
                notes=_decode_str(notes) if notes is not None else None,
            ))
    results: dict[str, dict[str, Any]] = {}
    if "results" in g:
        for result_id in g["results"].keys():
            payload = _read_payload_group(g["results"][result_id])
            results[str(result_id)] = payload if isinstance(payload, dict) else {}
    return PhotometryState(
        time_seconds=t,
        signals=s,
        channel_names=channel_names,
        history=h,
        summary=tuple(summary),
        derived=derived,
        results=results,
        metadata=metadata,
        readonly=bool(g.attrs.get("readonly", False)),
    )


def save_state_h5(state: PhotometryState, path: Path | str, *, compression: str | None = "gzip", compression_opts: int = 4) -> None:
    h5py = _require_h5py()
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "fibphot_state"
        f.attrs["schema_version"] = 2
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        _write_state_group(f, state, compression=compression, compression_opts=compression_opts)


def load_state_h5(path: Path | str) -> PhotometryState:
    h5py = _require_h5py()
    with h5py.File(Path(path), "r") as f:
        return _read_state_group(f)


def save_collection_h5(coll: Any, path: Path | str, *, compression: str | None = "gzip", compression_opts: int = 4) -> None:
    h5py = _require_h5py()
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "fibphot_collection"
        f.attrs["schema_version"] = 2
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
            _write_state_group(g_states.create_group(name), st, compression=compression, compression_opts=compression_opts)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("order", data=np.array(order, dtype=object), dtype=dt)


def load_collection_h5(path: Path | str):
    h5py = _require_h5py()
    from ..collection import PhotometryCollection
    with h5py.File(Path(path), "r") as f:
        order = [_decode_str(x) for x in f["order"][...]]
        states = [_read_state_group(f["states"][key]) for key in order]
    return PhotometryCollection.from_iterable(states)

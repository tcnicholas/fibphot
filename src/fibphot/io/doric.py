from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from ..state import PhotometryState
from ..types import FloatArray

AlignMode = Literal["truncate", "interp"]
Source = Literal[
    "lockin",       # demodulated photometry channels (GCaMP, Iso, etc.)
    "analog_in",    # raw detector voltages
    "analog_out"    # LED modulation outputs
]

@dataclass(frozen=True, slots=True)
class DoricChannel:
    name: str
    signal_path: str
    time_path: str
    attrs: dict[str, Any]

def _decode_attr(value: Any) -> Any:
    """ Best-effort decoding for HDF5 attributes. """

    if isinstance(value, bytes):
        return value.decode(errors="replace")

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O"}:
            out: list[Any] = []
            for v in value.ravel().tolist():
                out.append(_decode_attr(v))
            return np.array(out, dtype=object).reshape(value.shape)
        if value.size == 1:
            return _decode_attr(value.item())
        return value

    if isinstance(value, np.generic):
        return value.item()

    return value

def _get_attrs(obj: h5py.Dataset | h5py.Group) -> dict[str, Any]:
    """ Extract all attributes from an HDF5 group/dataset and decode them."""
    return {k: _decode_attr(v) for k, v in obj.attrs.items()}

def _normalise_name(name: str) -> str:
    """ Normalise channel names so they’re consistent keys. """
    return name.strip().lower().replace(" ", "_").replace("-", "_")

def _read_1d(dataset: h5py.Dataset) -> FloatArray:
    """ Read a dataset and ensure it’s a 1D float array. """
    arr = np.asarray(dataset[()], dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D dataset, got shape {arr.shape}.")
    return arr

def _discover_series_names(f: h5py.File, fpconsole: str) -> list[str]:
    """ Find which Doric “SeriesXXXX” groups exist in the file."""
    base = f"DataAcquisition/{fpconsole}/Signals"
    if base not in f:
        raise KeyError(f"Missing signals root: {base!r}")

    grp = f[base]
    series = [k for k in grp if str(k).lower().startswith("series")]
    if not series:
        raise ValueError(f"No series groups found under {base!r}")
    return sorted(series)

def _series_sort_key(name: str) -> tuple[int, str]:
    """ Sort key for series names like 'Series0001'. """
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else -1, name)

def _choose_series(series: str | None, available: list[str]) -> str:
    """ Decide which series to use. """
    if series is not None:
        if series not in available:
            raise KeyError(
                f"Series {series!r} not found. Available: {available}"
            )
        return series

    if len(available) == 1:
        return available[0]

    # Choose highest numbered series by default.
    return sorted(available, key=_series_sort_key)[-1]

def _find_series_root(f: h5py.File, fpconsole: str, series: str) -> str:
    """ Construct and validate the HDF5 path to the chosen series. """
    root = f"DataAcquisition/{fpconsole}/Signals/{series}"
    if root not in f:
        raise KeyError(f"Could not find series root: {root!r}")
    return root

def _available_sources(f: h5py.File, series_root: str) -> list[Source]:
    """ Discover which source types are available in the series. """
    out: list[Source] = []

    if f"{series_root}/AnalogIn" in f:
        out.append("analog_in")
    if f"{series_root}/AnalogOut" in f:
        out.append("analog_out")

    grp = f[series_root]
    if any(str(k).startswith("LockIn") for k in grp):
        out.append("lockin")

    order: list[Source] = ["lockin", "analog_in", "analog_out"]
    return [s for s in order if s in out]

def _choose_source(source: Source | None, available: list[Source]) -> Source:
    """
    Decide which source to use. 
    
    Priority: lockin > analog_in > analog_out. For photometry, lockin is 
    preferred as it contains demodulated signals.
    """
    if source is not None:
        if source not in available:
            raise KeyError(f"Source {source!r} not available. Available: {available}")
        return source

    # Priority: lockin > analog_in > analog_out
    for preferred in ("lockin", "analog_in", "analog_out"):
        if preferred in available:
            return preferred  # type: ignore[return-value]

    raise ValueError("No supported sources found in file.")

def _discover_lockin_channels(
    f: h5py.File,
    series_root: str
) -> list[DoricChannel]:
    """ Discover LockIn (demodulated) channels under the series root. """

    channels: list[DoricChannel] = []

    grp = f[series_root]
    lockin_keys = [k for k in grp if str(k).startswith("LockIn")]
    for key in lockin_keys:
        grp_path = f"{series_root}/{key}"
        lock_grp = f[grp_path]

        time_path = f"{grp_path}/Time"
        if time_path not in f:
            continue

        # Pick first non-Time dataset as signal (AIN01 is typical).
        dset_names = [k for k in lock_grp if str(k).lower() != "time"]
        if not dset_names:
            continue

        signal_path = (
            f"{grp_path}/AIN01" if f"{grp_path}/AIN01" in f 
            else f"{grp_path}/{dset_names[0]}"
        )

        ds = f[signal_path]
        attrs = _get_attrs(ds)

        username = attrs.get("Username")
        name = (
            str(username) if username not in (None, "", "0") 
            else ds.name.split("/")[-1]
        )
        channels.append(
            DoricChannel(
                name=_normalise_name(name),
                signal_path=signal_path,
                time_path=time_path,
                attrs=attrs,
            )
        )

    if not channels:
        raise ValueError(
            "No LockIn channels found. "
            f"Expected groups like 'LockInAOUT02' under {series_root!r}."
        )

    dedup: dict[str, DoricChannel] = {}
    for ch in channels:
        dedup.setdefault(ch.name, ch)

    return list(dedup.values())

def _discover_analog_in(f: h5py.File, series_root: str) -> list[DoricChannel]:
    """ Discover analogue input channels. """

    grp_path = f"{series_root}/AnalogIn"
    if grp_path not in f:
        raise KeyError(f"Missing group: {grp_path!r}")

    time_path = f"{grp_path}/Time"
    if time_path not in f:
        raise KeyError(f"Missing time dataset: {time_path!r}")

    grp = f[grp_path]
    channels: list[DoricChannel] = []
    for k in grp:
        if str(k).lower() == "time":
            continue
        ds_path = f"{grp_path}/{k}"
        ds = f[ds_path]
        attrs = _get_attrs(ds)
        username = attrs.get("Username")
        name = str(username) if username not in (None, "", "0") else str(k)
        channels.append(
            DoricChannel(
                name=_normalise_name(name),
                signal_path=ds_path,
                time_path=time_path,
                attrs=attrs,
            )
        )

    if not channels:
        raise ValueError(f"No analogue-in datasets found under {grp_path!r}")

    return channels

def _discover_analog_out(f: h5py.File, series_root: str) -> list[DoricChannel]:
    """ Discover analogue output channels (LED modulation outputs). """

    grp_path = f"{series_root}/AnalogOut"
    if grp_path not in f:
        raise KeyError(f"Missing group: {grp_path!r}")

    time_path = f"{grp_path}/Time"
    if time_path not in f:
        raise KeyError(f"Missing time dataset: {time_path!r}")

    grp = f[grp_path]
    channels: list[DoricChannel] = []
    for k in grp:
        if str(k).lower() == "time":
            continue
        ds_path = f"{grp_path}/{k}"
        ds = f[ds_path]
        attrs = _get_attrs(ds)
        username = attrs.get("Username")
        name = str(username) if username not in (None, "", "0") else str(k)
        channels.append(
            DoricChannel(
                name=_normalise_name(name),
                signal_path=ds_path,
                time_path=time_path,
                attrs=attrs,
            )
        )

    if not channels:
        raise ValueError(f"No analogue-out datasets found under {grp_path!r}")

    return channels

def _align_to_reference(
    x_ref: FloatArray,
    x: FloatArray,
    y: FloatArray,
    mode: AlignMode,
) -> FloatArray:
    """
    Align y(x) to the reference timebase x_ref.

    - truncate: cut to the shortest length (no resampling)
    - interp: interpolate y onto x_ref
    """

    if mode == "truncate":
        n = min(x_ref.shape[0], x.shape[0], y.shape[0])
        return np.asarray(y[:n], dtype=float)

    if mode == "interp":
        n = min(x.shape[0], y.shape[0])
        return np.interp(x_ref, x[:n], y[:n]).astype(float)

    raise ValueError(f"Unknown align mode: {mode!r}")

def read_doric(
    filename: Path | str,
    *,
    fpconsole: str = "FPConsole",
    series: str | None = None,
    source: Source | None = None,
    channels: Sequence[str] | Mapping[str, str] | None = None,
    align: AlignMode = "truncate",
) -> PhotometryState:
    """
    Read a Doric .doric (HDF5) file into a PhotometryState.

    Automatic behaviour
    -------------------
    - If series is None: choose only series if there is one, otherwise choose
      the highest-numbered series.
    - If source is None: prefer lockin > analog_in > analog_out.
    - Channel naming uses dataset attribute 'Username' when available.

    Parameters
    ----------
    fpconsole:
        Device group under DataAcquisition (usually 'FPConsole').
    series:
        Series group name like 'Series0001'. If None, auto-selected.
    source:
        'lockin', 'analog_in', or 'analog_out'. If None, auto-selected.
    channels:
        - None: load all discovered channels for the selected source
        - Sequence[str]: select by discovered channel names (normalised)
        - Mapping[str, str]: rename channels: {new_name: existing_name}
    align:
        - 'truncate': truncate all channels to the shortest timebase length
        - 'interp': interpolate all channels onto a reference timebase
    """

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, "r") as f:
        series_names = _discover_series_names(f, fpconsole)
        chosen_series = _choose_series(series, series_names)
        series_root = _find_series_root(f, fpconsole, chosen_series)

        available = _available_sources(f, series_root)
        chosen_source = _choose_source(source, available)

        if chosen_source == "lockin":
            discovered = _discover_lockin_channels(f, series_root)
        elif chosen_source == "analog_in":
            discovered = _discover_analog_in(f, series_root)
        else:
            discovered = _discover_analog_out(f, series_root)

        by_name = {c.name: c for c in discovered}

        if channels is None:
            selected = discovered
            out_names = [c.name for c in selected]
        elif isinstance(channels, Mapping):
            selected = []
            out_names = []
            for new_name, old_name in channels.items():
                old_key = _normalise_name(str(old_name))
                if old_key not in by_name:
                    raise KeyError(
                        f"Unknown channel {old_name!r}. "
                        f"Available: {sorted(by_name)}"
                    )
                selected.append(by_name[old_key])
                out_names.append(_normalise_name(str(new_name)))
        else:
            wanted = [_normalise_name(str(c)) for c in channels]
            missing = [c for c in wanted if c not in by_name]
            if missing:
                raise KeyError(
                    f"Unknown channel(s): {missing}. "
                    f"Available: {sorted(by_name)}"
                )
            selected = [by_name[c] for c in wanted]
            out_names = wanted

        if not selected:
            raise ValueError("No channels selected to load.")

        # Reference timebase: first selected channel.
        t_ref = _read_1d(f[selected[0].time_path])

        signals: list[FloatArray] = []
        for ch in selected:
            t = _read_1d(f[ch.time_path])
            y = _read_1d(f[ch.signal_path])
            signals.append(_align_to_reference(t_ref, t, y, mode=align))

        if align == "truncate":
            n = min([t_ref.shape[0], *[s.shape[0] for s in signals]])
            t_ref = t_ref[:n]
            signals = [s[:n] for s in signals]

        stacked = np.stack(signals, axis=0)

        meta: dict[str, Any] = {
            "file": str(path),
            "fpconsole": fpconsole,
            "series": chosen_series,
            "available_series": series_names,
            "source": chosen_source,
            "available_sources": available,
            "align": align,
            "channels": {
                name: by_name.get(name).attrs if name in by_name else {}
                for name in out_names
            },
        }

        gs_path = f"Configurations/{fpconsole}/GlobalSettings"
        ss_path = f"Configurations/{fpconsole}/SavingSettings"
        if gs_path in f:
            meta["global_settings"] = _get_attrs(f[gs_path])
        if ss_path in f:
            meta["saving_settings"] = _get_attrs(f[ss_path])

        return PhotometryState(
            time_seconds=t_ref,
            signals=stacked,
            channel_names=tuple(out_names),
            metadata=meta,
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.ndimage as ndi
import scipy.signal

from ..state import PhotometryState
from ..types import FloatArray
from .base import StageOutput, UpdateStage, _resolve_channels


def _hampel_1d(
    x: FloatArray,
    window_size: int,
    n_sigmas: float,
    *,
    mad_scale: float = 1.4826,
    mode: str = "reflect",
    match_edges: bool = True,
) -> FloatArray:
    """
    Fast Hampel filter using rolling medians.

    Parameters
    ----------
    mad_scale:
        Scale factor so MAD estimates standard deviation under Normal noise.
    mode:
        Padding strategy for the rolling median ('reflect', 'nearest', ...).
    match_edges:
        If True, applies "shrinking window" behaviour at first/last k samples.
    """

    if window_size < 3:
        raise ValueError("window_size must be >= 3.")
    if window_size % 2 == 0:
        window_size += 1

    x = np.asarray(x, dtype=float)
    n = int(x.shape[0])
    k = window_size // 2

    # Rolling median
    med = ndi.median_filter(x, size=window_size, mode=mode)

    # Rolling MAD = median(|x - med|)
    abs_dev = np.abs(x - med)
    mad = mad_scale * ndi.median_filter(abs_dev, size=window_size, mode=mode)

    out = x.copy()
    mask = (mad > 1e-12) & (abs_dev > (n_sigmas * mad))
    out[mask] = med[mask]

    if match_edges and k > 0 and n > 0:
        left = min(k, n)
        for i in range(left):
            lo = 0
            hi = min(n, i + k + 1)
            w = x[lo:hi]
            m = float(np.median(w))
            s = mad_scale * float(np.median(np.abs(w - m)))
            if s > 1e-12 and abs(x[i] - m) > n_sigmas * s:
                out[i] = m

        right_start = max(0, n - k)
        for i in range(right_start, n):
            lo = max(0, i - k)
            hi = n
            w = x[lo:hi]
            m = float(np.median(w))
            s = mad_scale * float(np.median(np.abs(w - m)))
            if s > 1e-12 and abs(x[i] - m) > n_sigmas * s:
                out[i] = m

    return out


@dataclass(frozen=True, slots=True)
class HampelFilter(UpdateStage):
    """
    Applies a Hampel filter to specified channels.

    This implementation is fast: it uses rolling medians (SciPy) rather than
    a Python loop over all samples.

    Parameters
    ----------
    window_size:
        Size of the moving window (forced odd and >= 3).
    n_sigmas:
        Threshold in units of (scaled) MAD.
    channels:
        "all", a channel name, a list of names, or None for all.
    mad_scale:
        Scale factor converting MAD to sigma under Normal noise.
    mode:
        Padding mode used by the rolling median.
    match_edges:
        If True, uses shrinking-window behaviour at the edges.

    Context
    -------
    The Hampel filter is a robust method for outlier detection and correction
    in time series data. It replaces outliers with the median of neighbouring
    values within a specified window, making it effective for removing transient 
    spikes or noise without significantly distorting the underlying signal.

    Compared to the `MedianFilter`, which replaces each point with the median of 
    its neighbours, the Hampel filter specifically targets outliers based on
    their deviation from the local median. Hence, it does not alter the signal 
    unless an outlier is detected.
    """

    name: str = field(default="hampel_filter", init=False)

    window_size: int = 11
    n_sigmas: float = 3.0
    channels: str | list[str] | None = None

    mad_scale: float = 1.4826
    mode: str = "reflect"
    match_edges: bool = True

    def _params_for_summary(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "n_sigmas": self.n_sigmas,
            "channels": self.channels if self.channels is not None else "all",
            "mad_scale": self.mad_scale,
            "mode": self.mode,
            "match_edges": self.match_edges,
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        idxs = _resolve_channels(state, self.channels)
        new = state.signals.copy()
        for i in idxs:
            new[i] = _hampel_1d(
                new[i],
                self.window_size,
                self.n_sigmas,
                mad_scale=self.mad_scale,
                mode=self.mode,
                match_edges=self.match_edges,
            )
        return StageOutput(signals=new)


@dataclass(frozen=True, slots=True)
class MedianFilter(UpdateStage):
    name: str = field(default="median_filter", init=False)
    kernel_size: int = 5
    channels: str | list[str] | None = None

    def _params_for_summary(self) -> dict[str, object]:
        return {
            "kernel_size": self.kernel_size,
            "channels": self.channels if self.channels is not None else "all",
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        k = self.kernel_size + (self.kernel_size % 2 == 0)
        idxs = _resolve_channels(state, self.channels)

        new = state.signals.copy()
        for i in idxs:
            new[i] = scipy.signal.medfilt(new[i], kernel_size=k)

        return StageOutput(signals=new)


@dataclass(frozen=True, slots=True)
class LowPassFilter(UpdateStage):
    """
    Applies a zero-phase low-pass Butterworth filter to specified channels.

    Parameters
    ----------
    critical_frequency : float
        The critical frequency (in Hz) for the low-pass filter. This is where 
        the filter begins to attenuate higher frequencies.
    order : int
        The order of the Butterworth filter. Higher order filters have a 
        steeper roll-off.
    sampling_rate : float | None
        The sampling rate (in Hz) of the input signals. If None, uses the 
        sampling rate from the PhotometryState.
    channels : str | list[str] | None
        The channels to which the filter should be applied. Can be "all", a 
        single channel name, or a list of channel names. If None, defaults to 
        "all".
    representation : Literal["sos", "ba"]
        The filter representation to use. "sos" for second-order sections 
        (numerically stable), or "ba" for (b, a) coefficients.

    Context
    ------- 
    Biosensor kinetics typically operate on slower (e.g., sub-second) timescales 
    relative to higher-frequency electrical noise. A low-pass filter keeps low
    frequencies and attenuates high frequencies.
    """

    name: str = field(default="low_pass_filter", init=False)
    critical_frequency: float = 10.0
    order: int = 2
    sampling_rate: float | None = None
    channels: str | list[str] | None = None
    representation: Literal["sos", "ba"] = "sos"

    def _params_for_summary(self) -> dict[str, object]:
        return {
            "critical_frequency": self.critical_frequency,
            "order": self.order,
            "sampling_rate": self.sampling_rate,
            "channels": self.channels if self.channels is not None else "all",
            "representation": self.representation,
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        fs = (
            state.sampling_rate if self.sampling_rate is None
            else float(self.sampling_rate)
        )
        if not (0.0 < self.critical_frequency < 0.5 * fs):
            raise ValueError(
                "critical_frequency must be > 0 and < Nyquist (fs/2). "
                f"Got critical_frequency={self.critical_frequency}, fs={fs}."
            )

        idxs = _resolve_channels(state, self.channels)
        new = state.signals.copy()

        if self.representation == "sos":
            sos = scipy.signal.butter(
                N=self.order,
                Wn=self.critical_frequency,
                btype="low",
                fs=fs,
                output="sos",
            )
            for i in idxs:
                new[i] = scipy.signal.sosfiltfilt(sos, new[i])
            return StageOutput(signals=new)

        if self.representation == "ba":

            res = scipy.signal.butter(
                N=self.order,
                Wn=self.critical_frequency,
                btype="low",
                fs=fs,
                output="ba",
            )

            if res is None:
                raise RuntimeError(
                    "scipy.signal.butter returned None; check filter params."
                )
            
            assert len(res)==2, "Expected (b,a) tuple from scipy.signal.butter."
            b, a = res

            for i in idxs:
                new[i] = scipy.signal.filtfilt(b, a, new[i])

            return StageOutput(signals=new)

        raise ValueError(f"Unknown representation: {self.representation!r}")
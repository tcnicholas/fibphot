from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..state import PhotometryState
from .base import StageOutput, UpdateStage, _resolve_channels

NormaliseMethod = Literal["baseline", "z_score", "null_z"]
BaselineMode = Literal["dff", "percent"]
NullZScale = Literal["rms", "mad"]


def _window_mask(
    state: PhotometryState,
    time_window: tuple[float, float] | None,
) -> np.ndarray:
    if time_window is None:
        return np.ones(state.n_samples, dtype=bool)

    t0, t1 = time_window
    if t1 < t0:
        raise ValueError("time_window must satisfy t0 <= t1.")

    mask = (state.time_seconds >= t0) & (state.time_seconds <= t1)
    if not np.any(mask):
        raise ValueError(
            f"time_window={time_window} selects no samples; check your time range."
        )
    return mask


@dataclass(frozen=True, slots=True)
class Normalise(UpdateStage):
    """
    Normalise photometry signals using one of several common schemes.

    Use the class constructors for clarity:

        Normalise.baseline(...)
        Normalise.z_score(...)
        Normalise.null_z(...)

    Notes
    -----
    This stage always operates on `state.signals` as they currently stand.
    For baseline normalisation, this typically means you should run motion
    correction first so your signals represent dF.
    """

    name: str = field(default="normalise", init=False)

    method: NormaliseMethod = "baseline"
    channels: str | list[str] | None = None

    # baseline normalisation
    baseline_key: str | None = "double_exp_baseline"
    baseline_mode: BaselineMode = "percent"

    # z-score / null-z window
    time_window: tuple[float, float] | None = None
    ddof: int = 0

    # null-z options
    null_z_scale: NullZScale = "rms"
    mad_scale: float = 1.4826

    # numerical safety
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.method == "baseline":
            if not self.baseline_key:
                raise ValueError(
                    "baseline_key must be set when method='baseline'."
                )
        else:
            # baseline parameters should not be used for non-baseline methods
            if self.baseline_key not in (None, "double_exp_baseline"):
                raise ValueError(
                    "baseline_key is only valid when method='baseline'. "
                    "Use Normalise.baseline(...)."
                )

        if self.ddof < 0:
            raise ValueError("ddof must be >= 0.")
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")
        if self.mad_scale <= 0:
            raise ValueError("mad_scale must be > 0.")
        if self.time_window is not None:
            t0, t1 = self.time_window
            if t1 < t0:
                raise ValueError("time_window must satisfy t0 <= t1.")

    @classmethod
    def baseline(
        cls,
        *,
        baseline_key: str = "double_exp_baseline",
        mode: BaselineMode = "percent",
        channels: str | list[str] | None = None,
        eps: float = 1e-12,
    ) -> Normalise:
        return cls(
            method="baseline",
            channels=channels,
            baseline_key=baseline_key,
            baseline_mode=mode,
            eps=eps,
        )

    @classmethod
    def z_score(
        cls,
        *,
        channels: str | list[str] | None = None,
        time_window: tuple[float, float] | None = None,
        ddof: int = 0,
        eps: float = 1e-12,
    ) -> Normalise:
        return cls(
            method="z_score",
            channels=channels,
            time_window=time_window,
            ddof=ddof,
            eps=eps,
            baseline_key=None,
        )

    @classmethod
    def null_z(
        cls,
        *,
        channels: str | list[str] | None = None,
        time_window: tuple[float, float] | None = None,
        scale: NullZScale = "rms",
        mad_scale: float = 1.4826,
        eps: float = 1e-12,
    ) -> Normalise:
        return cls(
            method="null_z",
            channels=channels,
            time_window=time_window,
            null_z_scale=scale,
            mad_scale=mad_scale,
            eps=eps,
            baseline_key=None,
        )

    def _params_for_summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "channels": self.channels if self.channels is not None else "all",
            "baseline_key": self.baseline_key,
            "baseline_mode": self.baseline_mode,
            "time_window": self.time_window,
            "ddof": self.ddof,
            "null_z_scale": self.null_z_scale,
            "mad_scale": self.mad_scale,
            "eps": self.eps,
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        idxs = _resolve_channels(state, self.channels)
        new = state.signals.copy()

        if self.method == "baseline":
            assert self.baseline_key is not None  # for type-checkers

            if self.baseline_key not in state.derived:
                raise KeyError(
                    f"Baseline '{self.baseline_key}' not found in state.derived. "
                    "Run the stage that produces this baseline first."
                )

            baseline = np.asarray(state.derived[self.baseline_key], dtype=float)
            if baseline.shape != state.signals.shape:
                raise ValueError(
                    f"Baseline shape {baseline.shape} does not match signals shape "
                    f"{state.signals.shape}."
                )

            scale = 100.0 if self.baseline_mode == "percent" else 1.0
            for i in idxs:
                denom = baseline[i]
                denom = np.where(np.abs(denom) < self.eps, np.nan, denom)
                new[i] = scale * (new[i] / denom)

            return StageOutput(
                signals=new,
                results={
                    "method": "baseline",
                    "baseline_key": self.baseline_key,
                    "baseline_mode": self.baseline_mode,
                    "channels_normalised": idxs,
                },
            )

        mask = _window_mask(state, self.time_window)

        if self.method == "z_score":
            means: dict[str, float] = {}
            stds: dict[str, float] = {}

            for i in idxs:
                x = new[i]
                mu = float(np.nanmean(x[mask]))
                sd = float(np.nanstd(x[mask], ddof=self.ddof))
                if not np.isfinite(sd) or sd < self.eps:
                    raise ValueError(
                        f"Standard deviation too small/invalid for channel "
                        f"'{state.channel_names[i]}': {sd}."
                    )
                new[i] = (x - mu) / sd
                means[state.channel_names[i]] = mu
                stds[state.channel_names[i]] = sd

            return StageOutput(
                signals=new,
                results={
                    "method": "z_score",
                    "means": means,
                    "stds": stds,
                    "time_window": self.time_window,
                    "ddof": self.ddof,
                },
            )

        # null_z
        scales: dict[str, float] = {}
        for i in idxs:
            x = new[i]
            xm = x[mask]

            if self.null_z_scale == "rms":
                s0 = float(np.sqrt(np.nanmean(xm * xm)))
            else:
                s0 = float(self.mad_scale * np.nanmedian(np.abs(xm)))

            if not np.isfinite(s0) or s0 < self.eps:
                raise ValueError(
                    f"Null-Z scale too small/invalid for channel "
                    f"'{state.channel_names[i]}': {s0}."
                )

            new[i] = x / s0
            scales[state.channel_names[i]] = s0

        return StageOutput(
            signals=new,
            results={
                "method": "null_z",
                "null_z_scale": self.null_z_scale,
                "scales": scales,
                "time_window": self.time_window,
            },
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..fit.regression import fit_irls, fit_ols
from ..state import PhotometryState
from .base import StageOutput, UpdateStage, _resolve_channels

RegressionMethod = Literal["ols", "irls_tukey", "irls_huber"]


@dataclass(frozen=True, slots=True)
class IsosbesticRegression(UpdateStage):
    """
    Regress a control channel (typically isosbestic) onto one or more channels.

    For each target channel y and control x, fit:

        y ≈ intercept + slope * x

    Output:
        dF = y - y_hat

    Also stores:
        derived["motion_fit"] = y_hat (per channel; shape matches signals)

    Notes
    -----
    `motion_fit` is a nuisance estimate used for subtraction/diagnostics. It is
    not necessarily suitable as a denominator for dF/F, especially if signals
    have been detrended (e.g. double exponential subtraction).
    """

    name: str = field(default="isosbestic_regression", init=False)

    control: str = "iso"
    channels: str | list[str] | None = None

    method: RegressionMethod = "irls_tukey"
    include_intercept: bool = True

    # IRLS settings
    tuning_constant: float = 4.685
    max_iter: int = 100
    tol: float = 1e-10
    store_weights: bool = False

    def _params_for_summary(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "channels": self.channels if self.channels is not None else "all",
            "method": self.method,
            "include_intercept": self.include_intercept,
            "tuning_constant": self.tuning_constant,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "store_weights": self.store_weights,
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        control_idx = state.idx(self.control)
        x = state.signals[control_idx]

        idxs = _resolve_channels(state, self.channels)
        idxs = [i for i in idxs if i != control_idx]
        if not idxs:
            raise ValueError(
                "No target channels remain after excluding the control channel."
            )

        new = state.signals.copy()
        motion_fit = np.full_like(state.signals, np.nan, dtype=float)

        per_channel: dict[str, dict[str, Any]] = {}
        r2s: list[float] = []

        for i in idxs:
            name = state.channel_names[i]
            y = state.signals[i]

            if self.method == "ols":
                fit = fit_ols(x, y, include_intercept=self.include_intercept)
                max_iter: int | None = None
            else:
                loss = "tukey" if self.method == "irls_tukey" else "huber"
                fit = fit_irls(
                    x,
                    y,
                    include_intercept=self.include_intercept,
                    loss=loss,
                    tuning_constant=self.tuning_constant,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    store_weights=self.store_weights,
                )
                max_iter = self.max_iter

            y_hat = fit.fitted
            motion_fit[i] = y_hat
            new[i] = y - y_hat

            per_channel[name] = {
                "control": self.control,
                "intercept": fit.intercept,
                "slope": fit.slope,
                "r2": fit.r2,
                "method": fit.method,
                "n_iter": fit.n_iter,
                "max_iter": max_iter,
                "tuning_constant": fit.tuning_constant,
                "scale": fit.scale,
                "weights": fit.weights,
            }
            if np.isfinite(fit.r2):
                r2s.append(float(fit.r2))

        metrics: dict[str, float] = {}
        if r2s:
            metrics["mean_r2"] = float(np.mean(r2s))
            metrics["median_r2"] = float(np.median(r2s))

        return StageOutput(
            signals=new,
            derived={
                "motion_fit": motion_fit,
            },
            results={
                "control": self.control,
                "control_idx": control_idx,
                "channels_fitted": idxs,
                "method": self.method,
                "include_intercept": self.include_intercept,
                "channels": per_channel,
            },
            metrics=metrics,
        )

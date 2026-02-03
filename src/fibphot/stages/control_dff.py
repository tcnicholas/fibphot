from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..fit.regression import fit_irls, fit_ols
from ..state import PhotometryState
from .base import StageOutput, UpdateStage, _resolve_channels

RegressionMethod = Literal["ols", "irls_tukey", "irls_huber"]
DffMode = Literal["dff", "percent", "df"]
DenomPolicy = Literal["nan", "clip", "raise"]


@dataclass(frozen=True, slots=True)
class IsosbesticDff(UpdateStage):
    """
    Combined control-fit normalisation workflow.

    Fits a control channel x (typically isosbestic) to each target channel y:

        y ≈ intercept + slope * x  = y_hat

    Then computes:
        df:     y - y_hat
        dff:    (y - y_hat) / y_hat
        percent: 100 * (y - y_hat) / y_hat

    This workflow is only numerically stable if y_hat does not spend meaningful
    time near zero (e.g. when fitting raw-ish, positive signals). If signals have
    already been detrended to be near zero (e.g. double-exp subtract=True),
    prefer IsosbesticRegression (df) + Normalise.baseline(double_exp_baseline).
    """

    name: str = field(default="isosbestic_dff", init=False)

    control: str = "iso"
    channels: str | list[str] | None = None

    method: RegressionMethod = "irls_tukey"
    include_intercept: bool = True

    mode: DffMode = "percent"

    # IRLS settings
    tuning_constant: float = 4.685
    max_iter: int = 100
    tol: float = 1e-10
    store_weights: bool = False

    # denominator safety
    min_abs_denom: float = 1e-6
    denom_policy: DenomPolicy = "nan"
    max_near_zero_frac: float = 0.01

    # optional outputs
    store_fit: bool = True
    store_df: bool = False

    def _params_for_summary(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "channels": self.channels if self.channels is not None else "all",
            "method": self.method,
            "include_intercept": self.include_intercept,
            "mode": self.mode,
            "tuning_constant": self.tuning_constant,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "store_weights": self.store_weights,
            "min_abs_denom": self.min_abs_denom,
            "denom_policy": self.denom_policy,
            "max_near_zero_frac": self.max_near_zero_frac,
            "store_fit": self.store_fit,
            "store_df": self.store_df,
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

        fit_mat = (
            np.full_like(state.signals, np.nan, dtype=float)
            if self.store_fit
            else None
        )
        df_mat = (
            np.full_like(state.signals, np.nan, dtype=float)
            if self.store_df
            else None
        )

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

            y_hat = np.asarray(fit.fitted, dtype=float)
            df = y - y_hat

            if fit_mat is not None:
                fit_mat[i] = y_hat
            if df_mat is not None:
                df_mat[i] = df

            if self.mode == "df":
                corrected = df
                denom_used = None
                near_zero_frac = 0.0
            else:
                abs_d = np.abs(y_hat)
                near_zero = abs_d < self.min_abs_denom
                near_zero_frac = float(np.mean(near_zero))

                if near_zero_frac > self.max_near_zero_frac:
                    msg = (
                        f"Denominator y_hat is near zero for {near_zero_frac:.2%} "
                        f"of samples in channel '{name}'. This usually indicates "
                        "you ran this after detrending/subtraction (signals ~ 0). "
                        "Use IsosbesticRegression (df) + Normalise.baseline(...) "
                        "instead, or run IsosbesticDff earlier on raw-ish signals."
                    )
                    if self.denom_policy == "raise":
                        raise ValueError(msg)

                if self.denom_policy == "nan":
                    denom = np.where(near_zero, np.nan, y_hat)
                else:
                    denom = np.where(
                        near_zero,
                        np.sign(y_hat) * self.min_abs_denom,
                        y_hat,
                    )

                scale = 100.0 if self.mode == "percent" else 1.0
                corrected = scale * (df / denom)
                denom_used = "y_hat"

            new[i] = corrected

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
                "denom_used": denom_used,
                "near_zero_frac": float(near_zero_frac),
                "min_abs_denom": float(self.min_abs_denom),
                "denom_policy": self.denom_policy,
            }
            if np.isfinite(fit.r2):
                r2s.append(float(fit.r2))

        metrics: dict[str, float] = {}
        if r2s:
            metrics["mean_r2"] = float(np.mean(r2s))
            metrics["median_r2"] = float(np.median(r2s))

        derived: dict[str, np.ndarray] = {}
        if fit_mat is not None:
            derived["control_fit"] = fit_mat
        if df_mat is not None:
            derived["control_df"] = df_mat

        return StageOutput(
            signals=new,
            derived=derived or None,
            results={
                "control": self.control,
                "control_idx": control_idx,
                "channels_fitted": idxs,
                "method": self.method,
                "mode": self.mode,
                "include_intercept": self.include_intercept,
                "channels": per_channel,
            },
            metrics=metrics,
        )

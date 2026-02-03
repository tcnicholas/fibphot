from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..types import FloatArray

RegressionMethod = Literal["ols", "irls_tukey", "irls_huber"]


@dataclass(frozen=True, slots=True)
class LinearFit:
    """
    Fit of y ≈ intercept + slope * x (or slope-only if include_intercept=False).
    """

    intercept: float
    slope: float
    fitted: FloatArray
    residuals: FloatArray
    r2: float
    method: RegressionMethod
    n_iter: int | None = None
    tuning_constant: float | None = None
    scale: float | None = None
    weights: FloatArray | None = None


def _r2_score(y: FloatArray, yhat: FloatArray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mad_sigma(x: FloatArray) -> float:
    """ Robust scale estimate using MAD, scaled for Normal data. """
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _design_matrix(x: FloatArray, include_intercept: bool) -> FloatArray:
    x = np.asarray(x, dtype=float)
    if include_intercept:
        return np.column_stack([np.ones_like(x), x])
    return x[:, None]


def fit_ols(
    x: FloatArray,
    y: FloatArray,
    *,
    include_intercept: bool = True,
) -> LinearFit:
    """
    Ordinary least squares fit of y on x.

    Context
    -------
    Applied to motion corrections in fiber photometry, x is the control signal
    (e.g., isosbestic channel) and y is the signal to be corrected. Hence, the
    estimated motion is given by:

        yhat = intercept + slope * x

    and the corrected signal is given by the residuals:

        corrected = y - yhat.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x0 = x[mask]
    y0 = y[mask]

    X = _design_matrix(x0, include_intercept)
    beta, *_ = np.linalg.lstsq(X, y0, rcond=None)

    if include_intercept:
        intercept = float(beta[0])
        slope = float(beta[1])
        yhat0 = intercept + slope * x0
    else:
        intercept = 0.0
        slope = float(beta[0])
        yhat0 = slope * x0

    fitted = np.full_like(y, np.nan, dtype=float)
    fitted[mask] = yhat0
    residuals = y - fitted

    return LinearFit(
        intercept=intercept,
        slope=slope,
        fitted=fitted,
        residuals=residuals,
        r2=_r2_score(y0, yhat0),
        method="ols",
    )


def _weights_tukey(u: FloatArray) -> FloatArray:
    """ Tukey's bisquare weights: w = (1 - u^2)^2 for |u|<1 else 0. """
    u = np.asarray(u, dtype=float)
    w = np.zeros_like(u)
    inside = np.abs(u) < 1.0
    w[inside] = (1.0 - u[inside] ** 2) ** 2
    return w


def _weights_huber(u: FloatArray) -> FloatArray:
    """ Huber weights: w = 1 for |u|<=1 else 1/|u|. """
    u = np.asarray(u, dtype=float)
    au = np.abs(u)
    w = np.ones_like(u)
    outside = au > 1.0
    w[outside] = 1.0 / au[outside]
    return w


def _wls_line(
    x: FloatArray,
    y: FloatArray,
    w: FloatArray,
    *,
    include_intercept: bool,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    w = np.clip(w, 0.0, np.inf)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 1e-15:
        return float("nan"), float("nan")

    if include_intercept:
        sx = float(np.sum(w * x))
        sy = float(np.sum(w * y))
        sxx = float(np.sum(w * x * x))
        sxy = float(np.sum(w * x * y))

        denom = sw * sxx - sx * sx
        if not np.isfinite(denom) or abs(denom) <= 1e-20:
            return float("nan"), float("nan")

        slope = (sw * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / sw
        return float(intercept), float(slope)

    # slope-only
    sxx = float(np.sum(w * x * x))
    if not np.isfinite(sxx) or sxx <= 1e-20:
        return 0.0, float("nan")

    sxy = float(np.sum(w * x * y))
    slope = sxy / sxx
    return 0.0, float(slope)


def fit_irls(
    x: FloatArray,
    y: FloatArray,
    *,
    include_intercept: bool = True,
    loss: Literal["tukey", "huber"] = "tukey",
    tuning_constant: float = 4.685,
    max_iter: int = 50,
    tol: float = 1e-10,
    store_weights: bool = False,
) -> LinearFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x0 = x[mask]
    y0 = y[mask]

    # initial OLS.
    if include_intercept:
        X = np.column_stack([np.ones_like(x0), x0])
        beta, *_ = np.linalg.lstsq(X, y0, rcond=None)
        intercept = float(beta[0])
        slope = float(beta[1])
    else:
        slope = float(np.dot(x0, y0) / (np.dot(x0, x0) + 1e-18))
        intercept = 0.0

    weight_fn = _weights_tukey if loss == "tukey" else _weights_huber

    w = np.ones_like(y0, dtype=float)
    scale: float | None = None

    n_iter = 0
    last_intercept = intercept
    last_slope = slope

    for _ in range(max_iter):
        n_iter += 1

        yhat = intercept + slope * x0 if include_intercept else slope * x0
        r = y0 - yhat

        scale = _mad_sigma(r)
        if not np.isfinite(scale) or scale <= 1e-15:
            break

        u = r / (tuning_constant * scale)
        w = weight_fn(u)

        if float(np.sum(w)) <= 1e-12:
            break

        intercept, slope = _wls_line(
            x0, y0, w, include_intercept=include_intercept
        )
        if not np.isfinite(slope) or (
            include_intercept and not np.isfinite(intercept)
        ):
            break

        # convergence
        di = abs(intercept - last_intercept) if include_intercept else 0.0
        ds = abs(slope - last_slope)
        denom = (
            abs(last_slope) + (
                abs(last_intercept) if include_intercept else 0.0
            ) + 1e-18
        )
        if (di + ds) / denom < tol:
            break

        last_intercept = intercept
        last_slope = slope

    yhat0 = intercept + slope * x0 if include_intercept else slope * x0

    fitted = np.full_like(y, np.nan, dtype=float)
    fitted[mask] = yhat0
    residuals = y - fitted

    weights_out = None
    if store_weights:
        weights_out = np.full_like(y, np.nan, dtype=float)
        weights_out[mask] = w

    method: RegressionMethod = "irls_tukey" if loss == "tukey" else "irls_huber"

    return LinearFit(
        intercept=float(intercept),
        slope=float(slope),
        fitted=fitted,
        residuals=residuals,
        r2=_r2_score(y0, yhat0),
        method=method,
        n_iter=n_iter,
        tuning_constant=float(tuning_constant),
        scale=float(scale) if scale is not None else None,
        weights=weights_out,
    )
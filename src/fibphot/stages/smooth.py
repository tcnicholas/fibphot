from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..state import PhotometryState
from ..types import FloatArray
from .base import StageOutput, UpdateStage, _resolve_channels

WindowType = Literal["flat", "hanning", "hamming", "bartlett", "blackman"]
PadMode = Literal["reflect", "edge"]
KalmanModel = Literal["local_level", "local_linear_trend"]


def _window(window: WindowType, window_len: int) -> FloatArray:
    if window == "flat":
        return np.ones(window_len, dtype=float)
    if window == "hanning":
        return np.hanning(window_len).astype(float)
    if window == "hamming":
        return np.hamming(window_len).astype(float)
    if window == "bartlett":
        return np.bartlett(window_len).astype(float)
    if window == "blackman":
        return np.blackman(window_len).astype(float)

    raise ValueError(
        "window must be one of: "
        "'flat', 'hanning', 'hamming', 'bartlett', 'blackman'."
    )


def smooth_1d(
    x: FloatArray,
    *,
    window_len: int,
    window: WindowType = "flat",
    pad_mode: PadMode = "reflect",
    match_edges: bool = True,
) -> FloatArray:
    """
    Convolution smoothing with explicit edge handling.

    Public helper so analysis code (e.g. peakfinding) can smooth *without*
    creating a new state / touching state.history.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("smooth_1d only accepts 1D arrays.")
    if window_len < 3:
        return x.copy()
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len % 2 == 0:
        raise ValueError("window_len must be odd.")

    w = _window(window, window_len)
    w = w / float(np.sum(w))

    half = window_len // 2

    if pad_mode == "reflect":
        xp = np.pad(x, pad_width=(half, half), mode="reflect")
    elif pad_mode == "edge":
        xp = np.pad(x, pad_width=(half, half), mode="edge")
    else:
        raise ValueError("pad_mode must be 'reflect' or 'edge'.")

    y = np.convolve(xp, w, mode="valid")

    if match_edges:
        y = y.copy()
        y[:half] = x[:half]
        y[-half:] = x[-half:]

    return y


def savgol_smooth_1d(
    x: FloatArray,
    *,
    window_len: int,
    polyorder: int = 3,
    mode: Literal["interp", "mirror", "nearest", "constant", "wrap"] = "interp",
) -> FloatArray:
    """
    Savitzky–Golay smoothing (shape-preserving; good for peaks).

    Requires scipy (already used elsewhere in fibphot).
    """
    from scipy.signal import savgol_filter

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("savgol_smooth_1d only accepts 1D arrays.")
    if window_len < 3:
        return x.copy()
    if window_len % 2 == 0:
        raise ValueError("window_len must be odd.")
    if polyorder >= window_len:
        raise ValueError("polyorder must be < window_len.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    # savgol_filter does not like NaNs; simple strategy:
    # interpolate NaNs linearly before filtering, then restore NaNs.
    m = np.isfinite(x)
    if not np.all(m):
        xi = np.arange(x.size, dtype=float)
        xp = np.interp(xi, xi[m], x[m])
        y = savgol_filter(xp, window_length=window_len, polyorder=polyorder, mode=mode)
        y[~m] = np.nan # type: ignore
        return np.asarray(y, dtype=float)

    return np.asarray(
        savgol_filter(x, window_length=window_len, polyorder=polyorder, mode=mode),
        dtype=float,
    )


def _mad_sigma(x: FloatArray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _estimate_obs_var_from_diff(y: FloatArray) -> float:
    """
    Robustly estimate observation variance R from first differences.

    If y_t = x_t + v_t with white noise v_t ~ N(0, R),
    then diff(y) has variance about 2R (ignoring process noise).
    """
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if np.sum(m) < 3:
        return float("nan")

    yy = y[m]
    d = np.diff(yy)
    s = _mad_sigma(d)  # robust std of diffs
    if not np.isfinite(s) or s <= 1e-20:
        return float("nan")

    # Var(diff) ~ 2R  => R ~ (s^2)/2
    return float((s * s) / 2.0)


def kalman_smooth_1d(
    y: FloatArray,
    *,
    model: KalmanModel = "local_level",
    r: float | Literal["auto"] = "auto",
    q: float | None = None,
    q_scale: float = 1e-3,
) -> FloatArray:
    """
    Fast Kalman RTS smoother (NumPy-only).

    model="local_level":
        x_t = x_{t-1} + w_t
        y_t = x_t + v_t

    model="local_linear_trend":
        [level, trend] evolves with small noise; smoother for slow drift.

    Parameters
    ----------
    r:
        Observation variance. "auto" estimates from first differences.
    q:
        Process variance (or process scale). If None, uses q = q_scale * r.
        Smaller q => smoother.
    q_scale:
        Used only if q is None.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if y.ndim != 1:
        raise ValueError("kalman_smooth_1d expects 1D input.")

    # observation variance
    if r == "auto":
        r_var = _estimate_obs_var_from_diff(y)
        if not np.isfinite(r_var) or r_var <= 1e-20:
            # fallback: variance of residual around median
            s = _mad_sigma(y)
            r_var = float(s * s) if np.isfinite(s) else 1e-6
    else:
        r_var = float(r)

    q_var = float(q_scale) * float(r_var) if q is None else float(q)

    if model == "local_level":
        # scalar filter
        x_f = np.full(n, np.nan, dtype=float)
        p_f = np.full(n, np.nan, dtype=float)
        x_p = np.full(n, np.nan, dtype=float)
        p_p = np.full(n, np.nan, dtype=float)

        # init from first finite sample
        m0 = np.isfinite(y)
        if not np.any(m0):
            return np.full_like(y, np.nan, dtype=float)
        i0 = int(np.argmax(m0))
        x = float(y[i0])
        p = float(r_var) * 10.0

        for t in range(n):
            # predict
            x_pred = x
            p_pred = p + q_var
            x_p[t] = x_pred
            p_p[t] = p_pred

            if np.isfinite(y[t]):
                # update
                k = p_pred / (p_pred + r_var)
                x = x_pred + k * (float(y[t]) - x_pred)
                p = (1.0 - k) * p_pred
            else:
                # missing obs: keep prediction
                x = x_pred
                p = p_pred

            x_f[t] = x
            p_f[t] = p

        # RTS smoother
        x_s = x_f.copy()
        p_s = p_f.copy()
        for t in range(n - 2, -1, -1):
            denom = p_p[t + 1]
            if not np.isfinite(denom) or denom <= 1e-30:
                continue
            c = p_f[t] / denom
            x_s[t] = x_f[t] + c * (x_s[t + 1] - x_p[t + 1])
            p_s[t] = p_f[t] + c * c * (p_s[t + 1] - p_p[t + 1])

        return x_s

    if model == "local_linear_trend":
        # 2D state: [level, trend]
        # level_t = level_{t-1} + trend_{t-1} + w1
        # trend_t = trend_{t-1} + w2
        # y_t = level_t + v
        x_f = np.full((n, 2), np.nan, dtype=float)
        p_f = np.full((n, 2, 2), np.nan, dtype=float)
        x_p = np.full((n, 2), np.nan, dtype=float)
        p_p = np.full((n, 2, 2), np.nan, dtype=float)

        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)

        # split q_var between level/trend
        Q = np.array([[q_var, 0.0], [0.0, q_var * 0.1]], dtype=float)
        R = np.array([[r_var]], dtype=float)

        m0 = np.isfinite(y)
        if not np.any(m0):
            return np.full_like(y, np.nan, dtype=float)
        i0 = int(np.argmax(m0))
        x = np.array([float(y[i0]), 0.0], dtype=float)
        P = np.eye(2, dtype=float) * float(r_var) * 10.0

        for t in range(n):
            # predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            x_p[t] = x_pred
            p_p[t] = P_pred

            if np.isfinite(y[t]):
                yt = np.array([[float(y[t])]], dtype=float)
                S = H @ P_pred @ H.T + R
                K = (P_pred @ H.T) @ np.linalg.inv(S)
                x = x_pred + (K @ (yt - (H @ x_pred).reshape(1, 1))).ravel()
                P = (np.eye(2) - K @ H) @ P_pred
            else:
                x = x_pred
                P = P_pred

            x_f[t] = x
            p_f[t] = P

        # RTS smoother
        x_s = x_f.copy()
        P_s = p_f.copy()
        for t in range(n - 2, -1, -1):
            P_pred_next = p_p[t + 1]
            if not np.all(np.isfinite(P_pred_next)):
                continue
            C = p_f[t] @ F.T @ np.linalg.inv(P_pred_next)
            x_s[t] = x_f[t] + C @ (x_s[t + 1] - x_p[t + 1])
            P_s[t] = p_f[t] + C @ (P_s[t + 1] - P_pred_next) @ C.T

        return x_s[:, 0]

    raise ValueError("model must be 'local_level' or 'local_linear_trend'.")


@dataclass(frozen=True, slots=True)
class Smooth(UpdateStage):
    """
    Smooth signals by convolving with a window function.
    """

    name: str = field(default="smooth", init=False)

    window_len: int = 11
    window: WindowType = "flat"
    pad_mode: PadMode = "reflect"
    match_edges: bool = True
    channels: str | list[str] | None = None

    def _params_for_summary(self) -> dict[str, Any]:
        return {
            "window_len": self.window_len,
            "window": self.window,
            "pad_mode": self.pad_mode,
            "match_edges": self.match_edges,
            "channels": self.channels if self.channels is not None else "all",
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        if self.window_len < 3:
            return StageOutput(
                signals=state.signals.copy(),
                notes="No-op: window_len < 3."
            )
        if self.window_len % 2 == 0:
            raise ValueError("window_len must be odd and >= 3.")

        idxs = _resolve_channels(state, self.channels)
        new = state.signals.copy()

        for i in idxs:
            new[i] = smooth_1d(
                new[i],
                window_len=self.window_len,
                window=self.window,
                pad_mode=self.pad_mode,
                match_edges=self.match_edges,
            )

        return StageOutput(signals=new)


@dataclass(frozen=True, slots=True)
class SavGolSmooth(UpdateStage):
    """
    Savitzky–Golay smoothing stage (shape-preserving).
    """

    name: str = field(default="savgol_smooth", init=False)

    window_len: int = 11
    polyorder: int = 3
    mode: Literal["interp", "mirror", "nearest", "constant", "wrap"] = "interp"
    channels: str | list[str] | None = None

    def _params_for_summary(self) -> dict[str, Any]:
        return {
            "window_len": self.window_len,
            "polyorder": self.polyorder,
            "mode": self.mode,
            "channels": self.channels if self.channels is not None else "all",
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        if self.window_len < 3:
            return StageOutput(signals=state.signals.copy(), notes="No-op: window_len < 3.")
        if self.window_len % 2 == 0:
            raise ValueError("window_len must be odd and >= 3.")
        if self.polyorder >= self.window_len:
            raise ValueError("polyorder must be < window_len.")

        idxs = _resolve_channels(state, self.channels)
        new = state.signals.copy()

        for i in idxs:
            new[i] = savgol_smooth_1d(
                new[i],
                window_len=self.window_len,
                polyorder=self.polyorder,
                mode=self.mode,
            )

        return StageOutput(signals=new)


@dataclass(frozen=True, slots=True)
class KalmanSmooth(UpdateStage):
    """
    Kalman RTS smoothing stage (fast, NumPy-only).

    Use this mainly as a *detector helper* (e.g. peakfinding).
    """

    name: str = field(default="kalman_smooth", init=False)

    model: KalmanModel = "local_level"
    r: float | Literal["auto"] = "auto"
    q: float | None = None
    q_scale: float = 1e-3
    channels: str | list[str] | None = None

    def _params_for_summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "r": self.r,
            "q": self.q,
            "q_scale": self.q_scale,
            "channels": self.channels if self.channels is not None else "all",
        }

    def apply(self, state: PhotometryState) -> StageOutput:
        idxs = _resolve_channels(state, self.channels)
        new = state.signals.copy()

        for i in idxs:
            new[i] = kalman_smooth_1d(
                new[i],
                model=self.model,
                r=self.r,
                q=self.q,
                q_scale=self.q_scale,
            )

        return StageOutput(signals=new)

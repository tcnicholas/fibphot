from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.optimize
import scipy.signal

from .types import FloatArray

if TYPE_CHECKING:
    import pandas as pd


PeakKind = Literal["peak", "valley"]
SelectKind = Literal["peak", "valley", "both"]
AreaRegion = Literal["bases", "fwhm"]
BaselineMode = Literal["line", "flat"]
FitModelName = Literal["gaussian", "lorentzian", "alpha"]


def _as_1d_float(x: FloatArray | None, n: int) -> FloatArray:
    if x is None:
        return np.arange(n, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.shape[0] != n:
        raise ValueError(f"x must be 1D with length {n}; got {x.shape}.")
    return x


def _interp_x_at_positions(x: FloatArray, positions: FloatArray) -> FloatArray:
    """
    Convert fractional sample positions (e.g. left_ips/right_ips) into x-units.
    """
    idx = np.arange(x.shape[0], dtype=float)
    return np.interp(positions, idx, x)


def _baseline_flat(y: FloatArray, left_i: int, right_i: int) -> float:
    """ Flat baseline between two indices. """
    return float(0.5 * (y[left_i] + y[right_i]))


def _baseline_line(
    x: FloatArray,
    y: FloatArray,
    left_i: int,
    right_i: int
) -> FloatArray:
    """ Linear baseline between two indices. """
    x0 = float(x[left_i])
    x1 = float(x[right_i])
    if np.isclose(x1, x0):
        return np.full_like(x, fill_value=float(y[left_i]), dtype=float)

    m = (float(y[right_i]) - float(y[left_i])) / (x1 - x0)
    c = float(y[left_i]) - m * x0
    return m * x + c


def _r2(y: FloatArray, yhat: FloatArray) -> float:
    """ Coefficient of determination. """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y: FloatArray, yhat: FloatArray) -> float:
    """ Root mean square error. """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def gaussian(
    x: FloatArray,
    amp: float,
    mu: float,
    sigma: float,
    offset: float
) -> FloatArray:
    """ Gaussian function. """
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-12)
    return offset + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lorentzian(
    x: FloatArray,
    amp: float,
    x0: float,
    gamma: float,
    offset: float
) -> FloatArray:
    """ Lorentzian function. """
    x = np.asarray(x, dtype=float)
    gamma = max(float(gamma), 1e-12)
    return offset + amp * (gamma**2) / ((x - x0) ** 2 + gamma**2)


def alpha_transient(
    x: FloatArray,
    amp: float,
    t0: float,
    tau: float,
    offset: float,
) -> FloatArray:
    """
    Simple alpha-like transient:
        offset + amp * ((t - t0)/tau) * exp(1 - (t - t0)/tau) for t >= t0
        offset otherwise

    This is a convenient, stable-ish phenomenological model for brief transients.
    """
    x = np.asarray(x, dtype=float)
    tau = max(float(tau), 1e-12)
    dt = (x - t0) / tau
    out = np.full_like(x, fill_value=offset, dtype=float)
    mask = x >= t0
    out[mask] = offset + amp * dt[mask] * np.exp(1.0 - dt[mask])
    return out


_MODEL_FUNCS: dict[FitModelName, Callable[..., FloatArray]] = {
    "gaussian": gaussian,
    "lorentzian": lorentzian,
    "alpha": alpha_transient,
}


@dataclass(frozen=True, slots=True)
class PeakFit:
    model: FitModelName
    params: tuple[float, ...]
    covariance: FloatArray | None
    r2: float
    rmse: float
    success: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class Peak:
    """
    A single detected extremum with derived measurements.

    Notes
    -----
    - `kind="peak"` refers to a local maximum.
    - `kind="valley"` refers to a local minimum.
    - Measurements are reported in x-units if x was provided, otherwise in samples.
    """

    kind: PeakKind
    index: int
    x: float
    y: float

    prominence: float | None = None
    left_base_index: int | None = None
    right_base_index: int | None = None

    height: float | None = None
    """Height relative to the baseline at the peak. Peaks tend to be positive, valleys negative."""

    fwhm: float | None = None
    """Full-width at half-maximum (or half-depth for valleys), in x-units/samples."""

    left_ip: float | None = None
    right_ip: float | None = None
    """Interpolated left/right positions (in x-units/samples) used for width computations."""

    area: float | None = None
    """Signed area under the (baseline-corrected) peak region. Peaks positive, valleys negative."""

    fit: PeakFit | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def amplitude(self) -> float | None:
        """Absolute height (useful when treating peaks and valleys uniformly)."""
        if self.height is None:
            return None
        return float(abs(self.height))

    def with_fit(self, fit: PeakFit) -> Peak:
        return replace(self, fit=fit)


@dataclass(frozen=True, slots=True)
class PeakSet:
    """
    Collection of peaks/valleys detected from a single 1D trace.
    """

    x: FloatArray
    y: FloatArray
    peaks: tuple[Peak, ...]
    kind: SelectKind
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.peaks)

    def indices(self, kind: SelectKind | None = None) -> list[int]:
        if kind is None or kind == "both":
            return [p.index for p in self.peaks]
        return [p.index for p in self.peaks if p.kind == kind]

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for p in self.peaks:
            rows.append(
                {
                    "kind": p.kind,
                    "index": p.index,
                    "x": p.x,
                    "y": p.y,
                    "prominence": p.prominence,
                    "height": p.height,
                    "fwhm": p.fwhm,
                    "left_ip": p.left_ip,
                    "right_ip": p.right_ip,
                    "left_base_index": p.left_base_index,
                    "right_base_index": p.right_base_index,
                    "area": p.area,
                    "fit_model": None if p.fit is None else p.fit.model,
                    "fit_r2": None if p.fit is None else p.fit.r2,
                    "fit_rmse": None if p.fit is None else p.fit.rmse,
                    "fit_success": None if p.fit is None else p.fit.success,
                }
            )
        return pd.DataFrame(rows)

    def plot(
        self,
        *,
        ax=None,
        show_bases: bool = False,
        annotate: bool = False,
        label: str | None = None,
    ):
        """
        Quick visualisation: signal + markers at detected peaks.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(dpi=150)
        else:
            fig = ax.figure

        ax.plot(self.x, self.y, label=label)

        xs = [p.x for p in self.peaks if p.kind == "peak"]
        ys = [p.y for p in self.peaks if p.kind == "peak"]
        if xs:
            ax.scatter(xs, ys, label="peaks")

        xs = [p.x for p in self.peaks if p.kind == "valley"]
        ys = [p.y for p in self.peaks if p.kind == "valley"]
        if xs:
            ax.scatter(xs, ys, label="valleys")

        if show_bases:
            for p in self.peaks:
                if p.left_base_index is None or p.right_base_index is None:
                    continue
                ax.scatter(
                    [self.x[p.left_base_index], self.x[p.right_base_index]],
                    [self.y[p.left_base_index], self.y[p.right_base_index]],
                    marker="x",
                )

        if annotate:
            for p in self.peaks:
                ax.annotate(
                    f"{p.kind}@{p.x:.3g}",
                    (p.x, p.y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

        ax.legend(frameon=False, fontsize=8)
        return fig, ax

    def fit_all(
        self,
        *,
        model: FitModelName = "gaussian",
        window: float | None = None,
        window_samples: int | None = None,
        maxfev: int = 5000,
    ) -> PeakSet:
        """
        Fit a model to each peak in the set.

        You can specify the fitting window as either:
          - `window` in x-units (seconds), or
          - `window_samples` in sample count.

        If neither is provided, defaults to a modest window around each peak
        based on its FWHM when available.
        """
        fitted: list[Peak] = []
        for p in self.peaks:
            fitted.append(
                fit_peak(
                    self.x,
                    self.y,
                    p,
                    model=model,
                    window=window,
                    window_samples=window_samples,
                    maxfev=maxfev,
                )
            )
        return replace(self, peaks=tuple(fitted))


def _fit_window_indices(
    x: FloatArray,
    peak: Peak,
    *,
    window: float | None,
    window_samples: int | None,
) -> tuple[int, int]:
    n = x.shape[0]
    i0 = peak.index

    if window_samples is not None:
        half = max(int(window_samples // 2), 1)
        lo = max(0, i0 - half)
        hi = min(n, i0 + half + 1)
        return lo, hi

    if window is not None:
        half = float(window) / 2.0
        lo_x = peak.x - half
        hi_x = peak.x + half
        lo = int(np.searchsorted(x, lo_x, side="left"))
        hi = int(np.searchsorted(x, hi_x, side="right"))
        return max(0, lo), min(n, hi)

    # default: use FWHM if available, otherwise a small fallback region
    if peak.fwhm is not None and np.isfinite(peak.fwhm):
        half = float(peak.fwhm) * 2.0
        lo_x = peak.x - half
        hi_x = peak.x + half
        lo = int(np.searchsorted(x, lo_x, side="left"))
        hi = int(np.searchsorted(x, hi_x, side="right"))
        return max(0, lo), min(n, hi)

    half = 25
    lo = max(0, i0 - half)
    hi = min(n, i0 + half + 1)
    return lo, hi


def fit_peak(
    x: FloatArray,
    y: FloatArray,
    peak: Peak,
    *,
    model: FitModelName = "gaussian",
    window: float | None = None,
    window_samples: int | None = None,
    maxfev: int = 5000,
) -> Peak:
    """
    Fit a parametric model to a single peak.

    Returns a new Peak with `.fit` populated. Failures are captured in the fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    func = _MODEL_FUNCS[model]
    lo, hi = _fit_window_indices(x, peak, window=window, window_samples=window_samples)

    xf = x[lo:hi]
    yf = y[lo:hi]

    if xf.size < 5:
        return peak.with_fit(
            PeakFit(
                model=model,
                params=(),
                covariance=None,
                r2=float("nan"),
                rmse=float("nan"),
                success=False,
                message="Not enough points in fitting window.",
            )
        )

    # guesses / bounds
    offset0 = float(np.median(yf))
    amp0 = float(peak.y - offset0)
    if peak.kind == "valley":
        # keep the sign consistent for the optimiser
        amp0 = float(peak.y - offset0)

    mu0 = float(peak.x)

    # a rough scale guess
    if peak.fwhm is not None and np.isfinite(peak.fwhm) and peak.fwhm > 0:
        sigma0 = float(peak.fwhm) / 2.355
        gamma0 = float(peak.fwhm) / 2.0
        tau0 = float(max(peak.fwhm, 1e-6))
    else:
        dx = float(np.median(np.diff(xf))) if xf.size > 2 else 1.0
        sigma0 = 3.0 * dx
        gamma0 = 3.0 * dx
        tau0 = 3.0 * dx

    try:
        if model == "gaussian":
            p0 = (amp0, mu0, sigma0, offset0)
            bounds = (
                (-np.inf, float(xf.min()), 1e-12, -np.inf),
                (np.inf, float(xf.max()), np.inf, np.inf),
            )
        elif model == "lorentzian":
            p0 = (amp0, mu0, gamma0, offset0)
            bounds = (
                (-np.inf, float(xf.min()), 1e-12, -np.inf),
                (np.inf, float(xf.max()), np.inf, np.inf),
            )
        else:  # alpha
            p0 = (amp0, mu0, tau0, offset0)
            bounds = (
                (-np.inf, float(xf.min()), 1e-12, -np.inf),
                (np.inf, float(xf.max()), np.inf, np.inf),
            )

        popt, pcov = scipy.optimize.curve_fit(
            f=func,
            xdata=xf,
            ydata=yf,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )
        yhat = func(xf, *popt)
        fit = PeakFit(
            model=model,
            params=tuple(float(v) for v in popt),
            covariance=np.asarray(pcov, dtype=float),
            r2=_r2(yf, yhat),
            rmse=_rmse(yf, yhat),
            success=True,
        )
    except Exception as exc:  # noqa: BLE001
        fit = PeakFit(
            model=model,
            params=(),
            covariance=None,
            r2=float("nan"),
            rmse=float("nan"),
            success=False,
            message=str(exc),
        )

    return peak.with_fit(fit)


@dataclass(frozen=True, slots=True)
class PeakFinder:
    """
    Robust peak/valley detection and analysis for 1D traces.
    """

    kind: SelectKind = "peak"

    # scipy.signal.find_peaks parameters.
    height: float | tuple[float, float] | None = None
    threshold: float | tuple[float, float] | None = None
    distance: int | None = None
    prominence: float | tuple[float, float] | None = None
    width: float | tuple[float, float] | None = None
    wlen: int | None = None
    plateau_size: int | tuple[int, int] | None = None

    rel_height: float = 0.5
    baseline_mode: BaselineMode = "line"
    area_region: AreaRegion = "bases"

    def fit(self, y: FloatArray, *, x: FloatArray | None = None) -> PeakSet:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1D.")

        x = _as_1d_float(x, y.shape[0])

        peaks: list[Peak] = []

        if self.kind in ("peak", "both"):
            peaks.extend(self._fit_one_kind(y, x, kind="peak"))
        if self.kind in ("valley", "both"):
            peaks.extend(self._fit_one_kind(y, x, kind="valley"))

        peaks.sort(key=lambda p: p.index)

        meta = {
            "kind": self.kind,
            "rel_height": self.rel_height,
            "baseline_mode": self.baseline_mode,
            "area_region": self.area_region,
        }
        return PeakSet(x=x, y=y, peaks=tuple(peaks), kind=self.kind, meta=meta)

    def _find_peaks(self, y: FloatArray) -> tuple[np.ndarray, dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "height": self.height,
            "threshold": self.threshold,
            "distance": self.distance,
            "prominence": self.prominence,
            "width": self.width,
            "wlen": self.wlen,
            "plateau_size": self.plateau_size,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        idx, props = scipy.signal.find_peaks(y, **kwargs)
        return idx, props

    def _fit_one_kind(self, y: FloatArray, x: FloatArray, *, kind: PeakKind) -> list[Peak]:
        if kind == "peak":
            y_work = y
            sign = 1.0
        else:
            y_work = -y
            sign = -1.0

        idx, _ = self._find_peaks(y_work)
        if idx.size == 0:
            return []

        # prominences and bases
        prom, left_bases, right_bases = scipy.signal.peak_prominences(
            y_work,
            idx,
            wlen=self.wlen,
        )

        # widths at rel_height (FWHM if rel_height=0.5)
        widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(
            y_work,
            idx,
            rel_height=self.rel_height,
            wlen=self.wlen,
        )

        # map fractional positions to x-units
        left_ip_x = _interp_x_at_positions(x, left_ips.astype(float))
        right_ip_x = _interp_x_at_positions(x, right_ips.astype(float))

        out: list[Peak] = []

        for j, i0 in enumerate(idx.tolist()):
            lb = int(left_bases[j])
            rb = int(right_bases[j])

            # baseline at peak
            if self.baseline_mode == "flat":
                base_at_peak = _baseline_flat(y, lb, rb)
                base_line = None
            else:
                base_line = _baseline_line(x, y, lb, rb)
                base_at_peak = float(base_line[i0])

            height = float(y[i0] - base_at_peak)

            # width in x-units
            fwhm = float(right_ip_x[j] - left_ip_x[j])

            # area under curve over chosen region
            if self.area_region == "fwhm":
                a_lo_x = float(left_ip_x[j])
                a_hi_x = float(right_ip_x[j])
                lo = int(np.searchsorted(x, a_lo_x, side="left"))
                hi = int(np.searchsorted(x, a_hi_x, side="right"))
                lo = max(0, lo)
                hi = min(x.shape[0], hi)
            else:  # bases
                lo = min(lb, rb)
                hi = max(lb, rb) + 1

            if hi - lo >= 2:
                xs = x[lo:hi]
                ys = y[lo:hi]
                if self.baseline_mode == "flat":
                    base = float(_baseline_flat(y, lb, rb))
                    area = float(np.trapezoid(ys - base, xs))
                else:
                    assert base_line is not None
                    area = float(np.trapezoid(ys - base_line[lo:hi], xs))
            else:
                area = float("nan")

            # convert prominence and width_height back to original orientation
            prominence = float(prom[j])  # always positive in y_work space

            # width_height is in y_work space; map back to y space
            # for peaks: y_work == y, so same; for valleys: y_work == -y
            width_height_y = float(sign * width_heights[j])

            out.append(
                Peak(
                    kind=kind,
                    index=int(i0),
                    x=float(x[i0]),
                    y=float(y[i0]),
                    prominence=prominence,
                    left_base_index=lb,
                    right_base_index=rb,
                    height=height,
                    fwhm=fwhm,
                    left_ip=float(left_ip_x[j]),
                    right_ip=float(right_ip_x[j]),
                    area=area,
                    meta={"width_height": width_height_y},
                )
            )

        return out

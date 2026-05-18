from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field
from typing import Any, Literal

from ..analysis import AnalysisWindow
from ..analysis.auc import AUC
from ..analysis.connectivity import (
    CrossCorrelationAnalysis,
    GrangerCausalityAnalysis,
    PeakAlignedConnectivityAnalysis,
)
from ..analysis.peak_aligned import PeakTriggeredAverage
from ..analysis.peaks import PeakAnalysis, PeaksByTemplate, biexponential_peak
from ..analysis.qc import QCAnalysis
from ..stages import (
    DoubleExpBaseline,
    HampelFilter,
    IsosbesticDff,
    IsosbesticRegression,
    LowPassFilter,
    MedianFilter,
    Normalise,
    PyBaselinesBaseline,
    Smooth,
    Trim,
)

try:  # optional newer stages in this codebase
    from ..stages import Crop, KalmanSmooth, SavGolSmooth
except Exception:  # pragma: no cover
    Crop = None  # type: ignore[assignment]
    KalmanSmooth = None  # type: ignore[assignment]
    SavGolSmooth = None  # type: ignore[assignment]

ParamKind = Literal[
    "str",
    "int",
    "float",
    "bool",
    "select",
    "channel",
    "channels",
    "window",
    "callable",
    "dict",
    "json",
]

_MISSING = MISSING


def _json_default(value: Any) -> Any:
    """Return a JSON-friendly default for small config values."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, AnalysisWindow):
        return value.as_dict()
    return value


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """GUI/API metadata for one configurable parameter.

    The core package does not need to depend on Panel or any GUI library. This
    light-weight schema is enough for the GUI to build controls, for pipeline
    JSON to stay serialisable, and for new objects to be registered without
    editing the GUI itself.
    """

    default: Any = None
    kind: ParamKind = "json"
    label: str | None = None
    help: str | None = None
    options: list[Any] | dict[str, Any] | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    allow_none: bool = False
    depends_on: str | None = None
    defaults_by_value: Mapping[Any, Any] | None = None


@dataclass(frozen=True, slots=True)
class RegisteredObject:
    key: str
    factory: Callable[..., Any]
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    label: str | None = None
    group: str | None = None
    description: str = ""
    normalise: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    result_kind: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or self.key

    def defaults(self) -> dict[str, Any]:
        return {name: _json_default(spec.default) for name, spec in self.parameters.items()}

    def create(self, params: Mapping[str, Any] | None = None) -> Any:
        p = self.defaults()
        p.update(dict(params or {}))
        if self.normalise is not None:
            p = self.normalise(p)
        return self.factory(**p)


class Registry:
    """Registry for stages or analyses.

    New objects can be added with ``register`` or the decorator form::

        STAGE_REGISTRY.register("MyStage", MyStage, parameters={...})

        @STAGE_REGISTRY.decorator("MyStage", parameters={...})
        class MyStage(UpdateStage):
            ...

    The GUI reads this registry only; it does not need to know the concrete
    stage/analysis classes.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, RegisteredObject] = {}

    def register(
        self,
        key: str,
        factory: Callable[..., Any],
        *,
        parameters: Mapping[str, ParameterSpec] | None = None,
        label: str | None = None,
        group: str | None = None,
        description: str = "",
        normalise: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        result_kind: str | None = None,
        replace: bool = False,
    ) -> RegisteredObject:
        if key in self._items and not replace:
            raise ValueError(f"Duplicate {self.kind} registry key: {key!r}")
        obj = RegisteredObject(
            key=key,
            factory=factory,
            parameters=dict(parameters or {}),
            label=label,
            group=group,
            description=description,
            normalise=normalise,
            result_kind=result_kind,
        )
        self._items[key] = obj
        return obj

    def decorator(self, key: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _decorate(factory: Callable[..., Any]) -> Callable[..., Any]:
            self.register(key, factory, **kwargs)
            return factory

        return _decorate

    def get(self, key: str) -> RegisteredObject:
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> list[str]:
        return sorted(self._items, key=lambda k: (self._items[k].group or "", self._items[k].display_name))

    def values(self) -> list[RegisteredObject]:
        return [self._items[k] for k in self.keys()]

    def options(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        seen: set[str] = set()
        for spec in self.values():
            label = spec.display_name
            if spec.group:
                label = f"{spec.group} / {label}"
            if label in seen:
                label = f"{label} ({spec.key})"
            seen.add(label)
            labels[label] = spec.key
        return labels

    def create(self, key: str, params: Mapping[str, Any] | None = None) -> Any:
        if key not in self._items:
            raise KeyError(f"Unknown {self.kind} {key!r}. Available: {self.keys()}")
        return self._items[key].create(params)

    def default_params(self, key: str) -> dict[str, Any]:
        return self.get(key).defaults()

    def auto_register_dataclass(
        self,
        key: str,
        factory: Callable[..., Any],
        *,
        label: str | None = None,
        group: str | None = None,
        description: str = "",
        overrides: Mapping[str, ParameterSpec] | None = None,
        exclude: set[str] | None = None,
        replace: bool = False,
    ) -> RegisteredObject:
        params = infer_parameter_specs(factory, overrides=overrides, exclude=exclude)
        return self.register(
            key,
            factory,
            label=label,
            group=group,
            description=description,
            parameters=params,
            replace=replace,
        )


STAGE_REGISTRY = Registry("stage")
ANALYSIS_REGISTRY = Registry("analysis")


def infer_parameter_specs(
    factory: Callable[..., Any],
    *,
    overrides: Mapping[str, ParameterSpec] | None = None,
    exclude: set[str] | None = None,
) -> dict[str, ParameterSpec]:
    """Infer conservative ParameterSpecs from a callable signature.

    This is intentionally basic: it gives extension authors a useful default,
    while ``overrides`` can improve labels, bounds and widget kinds.
    """
    exclude = set(exclude or {"name", "stage_plot", "checkpoint"})
    overrides = dict(overrides or {})
    params: dict[str, ParameterSpec] = {}
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return dict(overrides)

    for name, p in signature.parameters.items():
        if name in exclude or p.kind in {p.VAR_POSITIONAL, p.VAR_KEYWORD}:
            continue
        if name in overrides:
            params[name] = overrides[name]
            continue
        default = None if p.default is inspect._empty else p.default
        allow_none = default is None
        kind: ParamKind = "json"
        if isinstance(default, bool):
            kind = "bool"
        elif isinstance(default, int) and not isinstance(default, bool):
            kind = "int"
        elif isinstance(default, float):
            kind = "float"
        elif isinstance(default, str):
            kind = "str"
        params[name] = ParameterSpec(default=_json_default(default), kind=kind, allow_none=allow_none)
    for name, spec in overrides.items():
        params.setdefault(name, spec)
    return params


def _import_object(path: str) -> Any:
    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def _window_from_payload(payload: Any) -> AnalysisWindow | None:
    if payload is None or isinstance(payload, AnalysisWindow):
        return payload
    if isinstance(payload, Mapping):
        return AnalysisWindow(
            start=payload.get("start", 0.0),
            end=payload.get("end", 0.0),
            ref=payload.get("ref", "seconds"),
            label=payload.get("label"),
        )
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        return AnalysisWindow(payload[0], payload[1], "seconds")
    raise TypeError("window must be None, an AnalysisWindow, a mapping, or a two-item sequence.")


def _normalise_window_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    if "window" in out:
        out["window"] = _window_from_payload(out["window"])
    return out


def _normalise_template_params(params: dict[str, Any]) -> dict[str, Any]:
    out = _normalise_window_params(params)
    func = out.pop("template_func", None)
    if func is not None:
        out["func"] = _import_object(str(func))
    elif "func" not in out:
        out["func"] = biexponential_peak
    return out


TEMPLATE_DETECTOR_DEFAULTS: dict[str, Any] = {
    "template_params": {"tau_rise": 0.1, "tau_decay": 0.5},
    "template_duration": 0.6,
    "match_factor": 1.5,
    "refractory_s": 0.1,
    "align_to": "peak_top",
    "search_window_s": 1.0,
    "enforce_aligned_refractory": True,
    "min_peak_amp": 0.005,
}

PEAK_DETECTOR_DEFAULTS: dict[str, Any] = {
    "kind": "peak",
    "smooth_for_detection": True,
    "smooth_window_len": 25,
    "height": None,
    "prominence": None,
    "distance_s": 0.25,
    "width_s": 0.8,
    "auto_height_sigmas": 1.0,
    "auto_prominence_sigmas": 2.0,
    "fit_model": "alpha",
}

DETECTOR_PARAM_DEFAULTS: dict[str, dict[str, Any]] = {
    "template": TEMPLATE_DETECTOR_DEFAULTS,
    "peaks": PEAK_DETECTOR_DEFAULTS,
}


def _detector_params(default: str = "template") -> ParameterSpec:
    """ParameterSpec for detector_params controlled by a detector select widget."""
    return ParameterSpec(
        default=DETECTOR_PARAM_DEFAULTS[default],
        kind="dict",
        label="Detector parameters",
        depends_on="detector",
        defaults_by_value=DETECTOR_PARAM_DEFAULTS,
        help=(
            "These parameters change automatically when Peak detector is changed. "
            "Template uses matched-filter settings; peaks uses local-maximum detection settings."
        ),
    )


def _stage_common_channels(default: Any = "all") -> ParameterSpec:
    return ParameterSpec(
        default=default,
        kind="channels",
        label="Channels",
        help=(
            "Choose one or more loaded channels. Once a recording is loaded, "
            "this defaults to all non-isosbestic channels."
        ),
    )


def _signal(default: str | None = None) -> ParameterSpec:
    return ParameterSpec(default=default, kind="channel", label="Signal", allow_none=default is None)


def _window(default: dict[str, Any] | None) -> ParameterSpec:
    return ParameterSpec(
        default=default,
        kind="window",
        label="Window",
        help="JSON object, e.g. {\"start\": 0, \"end\": 10, \"ref\": \"seconds\"}; use null for whole trace.",
        allow_none=True,
    )


def register_defaults(*, replace: bool = False) -> None:
    """Register built-in stages and analyses.

    This function is idempotent. Extension packages can import
    ``STAGE_REGISTRY``/``ANALYSIS_REGISTRY`` and register additional objects
    without any GUI edits.
    """
    if STAGE_REGISTRY.keys() and ANALYSIS_REGISTRY.keys() and not replace:
        return

    # Stages -----------------------------------------------------------------
    STAGE_REGISTRY.register(
        "Trim",
        Trim,
        label="Trim recording ends",
        group="Time",
        description="Discard time from the start and/or end of the recording.",
        parameters={
            "start": ParameterSpec(0.0, "float", label="Remove from start", minimum=0, step=1.0),
            "end": ParameterSpec(0.0, "float", label="Remove from end", minimum=0, step=1.0),
            "unit": ParameterSpec("seconds", "select", label="Unit", options=["seconds", "samples"]),
        },
        replace=replace,
    )
    if Crop is not None:
        STAGE_REGISTRY.register(
            "Crop",
            Crop,
            label="Crop to interval",
            group="Time",
            description="Keep only samples between start and stop.",
            parameters={
                "start": ParameterSpec(0.0, "float", label="Start", minimum=0, step=1.0),
                "stop": ParameterSpec(None, "float", label="Stop", minimum=0, step=1.0, allow_none=True),
                "unit": ParameterSpec("seconds", "select", label="Unit", options=["seconds", "samples"]),
            },
            replace=replace,
        )
    STAGE_REGISTRY.register(
        "HampelFilter",
        HampelFilter,
        label="Hampel filter",
        group="Filtering",
        description="Robust spike/outlier removal.",
        parameters={
            "window_size": ParameterSpec(13, "int", label="Window size", minimum=3, step=2),
            "n_sigmas": ParameterSpec(3.0, "float", label="Sigma threshold", minimum=0, step=0.1),
            "channels": _stage_common_channels(),
            "mad_scale": ParameterSpec(1.4826, "float", label="MAD scale", minimum=0, step=0.001),
            "mode": ParameterSpec("reflect", "select", label="Padding mode", options=["reflect", "nearest", "mirror", "wrap", "constant"]),
            "match_edges": ParameterSpec(True, "bool", label="Match edges"),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "LowPassFilter",
        LowPassFilter,
        label="Low-pass filter",
        group="Filtering",
        description="Butterworth low-pass filter.",
        parameters={
            "critical_frequency": ParameterSpec(10.0, "float", label="Cut-off frequency / Hz", minimum=0.001, step=0.1),
            "order": ParameterSpec(2, "int", label="Order", minimum=1, maximum=12, step=1),
            "sampling_rate": ParameterSpec(None, "float", label="Sampling rate override", minimum=0, step=1.0, allow_none=True),
            "channels": _stage_common_channels(),
            "representation": ParameterSpec("sos", "select", label="Representation", options=["sos", "ba"]),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "MedianFilter",
        MedianFilter,
        label="Median filter",
        group="Filtering",
        parameters={
            "kernel_size": ParameterSpec(5, "int", label="Kernel size", minimum=1, step=2),
            "channels": _stage_common_channels(),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "Smooth",
        Smooth,
        label="Window smoothing",
        group="Smoothing",
        parameters={
            "window_len": ParameterSpec(11, "int", label="Window length", minimum=3, step=2),
            "window": ParameterSpec("flat", "select", label="Window", options=["flat", "hanning", "hamming", "bartlett", "blackman"]),
            "pad_mode": ParameterSpec("reflect", "select", label="Padding mode", options=["reflect", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "symmetric", "wrap"]),
            "match_edges": ParameterSpec(True, "bool", label="Match edges"),
            "channels": _stage_common_channels(),
        },
        replace=replace,
    )
    if SavGolSmooth is not None:
        STAGE_REGISTRY.register(
            "SavGolSmooth",
            SavGolSmooth,
            label="Savitzky–Golay smoothing",
            group="Smoothing",
            parameters={
                "window_len": ParameterSpec(11, "int", label="Window length", minimum=3, step=2),
                "polyorder": ParameterSpec(3, "int", label="Polynomial order", minimum=0, step=1),
                "mode": ParameterSpec("interp", "select", label="Mode", options=["interp", "mirror", "nearest", "constant", "wrap"]),
                "channels": _stage_common_channels(),
            },
            replace=replace,
        )
    if KalmanSmooth is not None:
        STAGE_REGISTRY.register(
            "KalmanSmooth",
            KalmanSmooth,
            label="Kalman smoothing",
            group="Smoothing",
            parameters={
                "model": ParameterSpec("local_level", "select", label="Model", options=["local_level"]),
                "r": ParameterSpec("auto", "json", label="Observation variance", help="Use \"auto\" or a number."),
                "q": ParameterSpec(None, "float", label="Process variance", allow_none=True, minimum=0, step=0.001),
                "q_scale": ParameterSpec(1e-3, "float", label="Auto q scale", minimum=0, step=0.0001),
                "channels": _stage_common_channels(),
            },
            replace=replace,
        )
    STAGE_REGISTRY.register(
        "DoubleExpBaseline",
        DoubleExpBaseline,
        label="Double-exponential baseline",
        group="Baseline",
        parameters={
            "subtract": ParameterSpec(True, "bool", label="Subtract baseline"),
            "channels": _stage_common_channels(),
            "decimate_to_hz": ParameterSpec(None, "float", label="Fit decimation / Hz", allow_none=True, minimum=0, step=1.0),
            "maxfev": ParameterSpec(2000, "int", label="Max function evaluations", minimum=100, step=100),
            "tau_fast_bounds": ParameterSpec([60.0, 600.0], "json", label="Fast tau bounds"),
            "tau_slow_bounds": ParameterSpec([600.0, 36000.0], "json", label="Slow tau bounds"),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "PyBaselinesBaseline",
        PyBaselinesBaseline,
        label="pybaselines baseline",
        group="Baseline",
        parameters={
            "method": ParameterSpec("asls", "str", label="Method"),
            "method_kwargs": ParameterSpec({}, "dict", label="Method kwargs"),
            "channels": _stage_common_channels(),
            "x_axis": ParameterSpec("time", "select", label="x-axis", options=["time", "index"]),
            "baseline_key": ParameterSpec(None, "str", label="Baseline key", allow_none=True),
            "subtract": ParameterSpec(True, "bool", label="Subtract baseline"),
            "store_full_params": ParameterSpec(False, "bool", label="Store full params"),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "IsosbesticRegression",
        IsosbesticRegression,
        label="Isosbestic regression",
        group="Control correction",
        parameters={
            "control": ParameterSpec(None, "channel", label="Control channel", allow_none=True, help="The GUI will try to select the channel whose name is closest to 'iso'."),
            "channels": _stage_common_channels(None),
            "method": ParameterSpec("irls_tukey", "select", label="Method", options=["ols", "huber", "theil_sen", "irls_huber", "irls_tukey"]),
            "include_intercept": ParameterSpec(True, "bool", label="Include intercept"),
            "tuning_constant": ParameterSpec(1.4, "float", label="Tuning constant", minimum=0, step=0.1),
            "max_iter": ParameterSpec(100, "int", label="Max iterations", minimum=1, step=10),
            "tol": ParameterSpec(1e-10, "float", label="Tolerance", minimum=0, step=1e-10),
            "store_weights": ParameterSpec(False, "bool", label="Store weights"),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "IsosbesticDff",
        IsosbesticDff,
        label="Isosbestic dF/F",
        group="Control correction",
        parameters={
            "control": ParameterSpec(None, "channel", label="Control channel", allow_none=True, help="The GUI will try to select the channel whose name is closest to 'iso'."),
            "channels": _stage_common_channels(None),
            "method": ParameterSpec("irls_tukey", "select", label="Method", options=["ols", "huber", "theil_sen", "irls_huber", "irls_tukey"]),
            "include_intercept": ParameterSpec(True, "bool", label="Include intercept"),
            "mode": ParameterSpec("percent", "select", label="Output mode", options=["df", "dff", "percent"]),
            "tuning_constant": ParameterSpec(4.685, "float", label="Tuning constant", minimum=0, step=0.1),
            "store_weights": ParameterSpec(False, "bool", label="Store weights"),
            "store_fit": ParameterSpec(True, "bool", label="Store fit"),
            "store_df": ParameterSpec(False, "bool", label="Store ΔF"),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "Normalise.baseline",
        Normalise.baseline,
        label="Baseline normalisation",
        group="Normalisation",
        parameters={
            "baseline_key": ParameterSpec("double_exp_baseline", "str", label="Baseline key"),
            "mode": ParameterSpec("percent", "select", label="Mode", options=["df", "dff", "percent"]),
            "channels": _stage_common_channels(),
            "eps": ParameterSpec(1e-12, "float", label="Epsilon", minimum=0, step=1e-12),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "Normalise.z_score",
        Normalise.z_score,
        label="Z-score",
        group="Normalisation",
        parameters={
            "channels": _stage_common_channels(),
            "time_window": ParameterSpec(None, "json", label="Reference time window", allow_none=True, help="null or [start, end] in seconds."),
            "ddof": ParameterSpec(0, "int", label="ddof", minimum=0, step=1),
            "eps": ParameterSpec(1e-12, "float", label="Epsilon", minimum=0, step=1e-12),
        },
        replace=replace,
    )
    STAGE_REGISTRY.register(
        "Normalise.null_z",
        Normalise.null_z,
        label="Null-z score",
        group="Normalisation",
        parameters={
            "channels": _stage_common_channels(),
            "time_window": ParameterSpec(None, "json", label="Reference time window", allow_none=True),
            "scale": ParameterSpec("rms", "select", label="Scale", options=["rms", "mad"]),
            "mad_scale": ParameterSpec(1.4826, "float", label="MAD scale", minimum=0, step=0.001),
            "eps": ParameterSpec(1e-12, "float", label="Epsilon", minimum=0, step=1e-12),
        },
        replace=replace,
    )

    # Analyses ---------------------------------------------------------------
    ANALYSIS_REGISTRY.register(
        "QCAnalysis",
        QCAnalysis,
        label="QC metrics",
        group="Quality control",
        result_kind="summary",
        description="Basic quality-control metrics.",
        parameters={
            "signal": _signal(None),
            "saturation_low": ParameterSpec(None, "float", label="Saturation low", allow_none=True),
            "saturation_high": ParameterSpec(None, "float", label="Saturation high", allow_none=True),
        },
        replace=replace,
    )
    ANALYSIS_REGISTRY.register(
        "AUC",
        AUC,
        label="Area under curve",
        group="Trace metrics",
        result_kind="arrays",
        normalise=_normalise_window_params,
        parameters={
            "signal": _signal("gcamp"),
            "window": _window({"start": 0.0, "end": 10.0, "ref": "seconds", "label": None}),
            "mode": ParameterSpec("positive", "select", label="Mode", options=["signed", "positive", "negative", "absolute"]),
            "baseline": ParameterSpec("pre_median", "select", label="Baseline", options=["zero", "window_mean", "window_median", "pre_mean", "pre_median", "window_quantile", "pre_quantile"]),
            "pre_seconds": ParameterSpec(10.0, "float", label="Pre-window / s", minimum=0, step=1.0),
            "quantile": ParameterSpec(0.1, "float", label="Quantile", minimum=0, maximum=1, step=0.05),
        },
        replace=replace,
    )
    ANALYSIS_REGISTRY.register(
        "PeakAnalysis",
        PeakAnalysis,
        label="Peaks by local maxima",
        group="Peaks",
        result_kind="events",
        normalise=_normalise_window_params,
        parameters={
            "signal": _signal("gcamp"),
            "kind": ParameterSpec("peak", "select", label="Kind", options=["peak", "valley", "both"]),
            "window": _window(None),
            "smooth_for_detection": ParameterSpec(True, "bool", label="Smooth before detection"),
            "smooth_window_len": ParameterSpec(25, "int", label="Smooth window", minimum=3, step=2),
            "height": ParameterSpec(None, "json", label="Height", allow_none=True),
            "prominence": ParameterSpec(None, "json", label="Prominence", allow_none=True),
            "distance_s": ParameterSpec(0.25, "float", label="Distance / s", allow_none=True, minimum=0, step=0.05),
            "width_s": ParameterSpec(0.8, "float", label="Width / s", allow_none=True, minimum=0, step=0.05),
            "auto_height_sigmas": ParameterSpec(1.0, "float", label="Auto height sigmas", minimum=0, step=0.1),
            "auto_prominence_sigmas": ParameterSpec(2.0, "float", label="Auto prominence sigmas", minimum=0, step=0.1),
            "fit_model": ParameterSpec("alpha", "select", label="Fit model", allow_none=True, options=[None, "gaussian", "lorentzian", "alpha"]),
        },
        replace=replace,
    )
    ANALYSIS_REGISTRY.register(
        "PeaksByTemplate",
        PeaksByTemplate,
        label="Peaks by template",
        group="Peaks",
        result_kind="events",
        normalise=_normalise_template_params,
        parameters={
            "signal": _signal("rgeco"),
            "template_func": ParameterSpec("fibphot.analysis.peaks:biexponential_peak", "callable", label="Template function"),
            "template_params": ParameterSpec({"tau_rise": 0.1, "tau_decay": 0.5}, "dict", label="Template parameters"),
            "window": _window({"start": 0.0, "end": 1750.0, "ref": "seconds", "label": None}),
            "kind": ParameterSpec("peak", "select", label="Kind", options=["peak", "valley"]),
            "template_duration": ParameterSpec(0.6, "float", label="Template duration / s", minimum=0.001, step=0.05),
            "match_factor": ParameterSpec(1.5, "float", label="Match factor", minimum=0, step=0.1),
            "threshold": ParameterSpec(None, "float", label="Absolute threshold", allow_none=True),
            "refractory_s": ParameterSpec(0.1, "float", label="Refractory / s", minimum=0, step=0.05),
            "align_to": ParameterSpec("peak_top", "select", label="Align to", options=["peak_top", "match"]),
            "search_window_s": ParameterSpec(1.0, "float", label="Peak-top search span / s", minimum=0, step=0.05),
            "enforce_aligned_refractory": ParameterSpec(True, "bool", label="Suppress aligned duplicates"),
            "min_peak_amp": ParameterSpec(0.005, "float", label="Minimum peak amplitude", allow_none=True, step=0.001),
            "measure_widths": ParameterSpec(True, "bool", label="Measure widths"),
            "rel_height": ParameterSpec(0.5, "float", label="Relative width height", minimum=0, maximum=1, step=0.05),
            "fit_model": ParameterSpec(None, "select", label="Fit model", allow_none=True, options=[None, "gaussian", "lorentzian", "alpha"]),
        },
        replace=replace,
    )

    ANALYSIS_REGISTRY.register(
        "PeakTriggeredAverage",
        PeakTriggeredAverage,
        label="Peak-triggered average",
        group="Session averages",
        result_kind="aligned",
        normalise=_normalise_window_params,
        parameters={
            "event_signal": _signal("rgeco"),
            "channels": _stage_common_channels("all"),
            "detector": ParameterSpec("template", "select", label="Peak detector", options=["template", "peaks"]),
            "detector_params": _detector_params("template"),
            "window": _window(None),
            "t_before_s": ParameterSpec(20.0, "float", label="Before event / s", minimum=0, step=1.0),
            "t_after_s": ParameterSpec(20.0, "float", label="After event / s", minimum=0, step=1.0),
            "target_fs": ParameterSpec(None, "float", label="Target sampling / Hz", allow_none=True, minimum=0, step=1.0),
            "exclude_times": ParameterSpec([], "json", label="Exclude windows"),
            "exclude_mode": ParameterSpec("peak_time", "select", label="Exclude mode", options=["peak_time", "window_overlap"]),
            "min_previous_interval_s": ParameterSpec(None, "float", label="Minimum previous-event gap / s", allow_none=True, minimum=0, step=1.0),
            "baseline_window_s": ParameterSpec(None, "json", label="Baseline window relative to event", allow_none=True),
        },
        replace=replace,
    )

    ANALYSIS_REGISTRY.register(
        "CrossCorrelationAnalysis",
        CrossCorrelationAnalysis,
        label="Cross-correlation",
        group="Connectivity",
        result_kind="curve",
        normalise=_normalise_window_params,
        parameters={
            "x_signal": _signal("gcamp"),
            "y_signal": _signal("rgeco"),
            "window": _window(None),
            "max_lag_s": ParameterSpec(10.0, "float", label="Maximum lag / s", minimum=0, step=1.0),
            "detrend": ParameterSpec(True, "bool", label="Detrend"),
            "normalise": ParameterSpec(True, "bool", label="Normalise"),
        },
        replace=replace,
    )

    ANALYSIS_REGISTRY.register(
        "GrangerCausalityAnalysis",
        GrangerCausalityAnalysis,
        label="Granger causality",
        group="Connectivity",
        result_kind="curve",
        normalise=_normalise_window_params,
        parameters={
            "x_signal": _signal("gcamp"),
            "y_signal": _signal("rgeco"),
            "window": _window(None),
            "max_lag_s": ParameterSpec(5.0, "float", label="Maximum lag / s", minimum=0, step=1.0),
            "target_dt": ParameterSpec(1.0, "float", label="Target dt / s", allow_none=True, minimum=0, step=0.1, help="Downsample before Granger. Larger values are faster."),
            "max_lag_steps": ParameterSpec(20, "int", label="Maximum lag steps", allow_none=True, minimum=1, step=1, help="Caps the number of VAR lags tested."),
            "max_samples": ParameterSpec(2000, "int", label="Maximum samples", allow_none=True, minimum=100, step=100, help="Caps samples after downsampling for interactive use."),
            "detrend": ParameterSpec(True, "bool", label="Detrend"),
            "difference": ParameterSpec(True, "bool", label="Difference"),
            "fdr": ParameterSpec(True, "bool", label="FDR correction"),
        },
        replace=replace,
    )

    ANALYSIS_REGISTRY.register(
        "PeakAlignedConnectivityAnalysis",
        PeakAlignedConnectivityAnalysis,
        label="Peak-aligned connectivity",
        group="Connectivity",
        result_kind="aligned_connectivity",
        normalise=_normalise_window_params,
        description=(
            "Detect peaks in the event signal, align all selected channels to those "
            "event times, then compute event-aligned averages and optional "
            "connectivity curves. Typical use: align green peaks and inspect the "
            "mean ± SEM green and red responses in the same peri-event window."
        ),
        parameters={
            "event_signal": _signal("gcamp"),
            "channels": ParameterSpec(
                "all",
                "channels",
                label="Aligned channels",
                help=(
                    "Choose one or more loaded channels to extract around event-signal peaks. "
                    "Defaults to all non-isosbestic channels; the event signal is always included."
                ),
            ),
            "connectivity_pairs": ParameterSpec(None, "json", label="Connectivity pairs", allow_none=True, help='Optional list such as [["gcamp", "rgeco"]]. If null, event_signal is paired with every other aligned channel.'),
            "x_signal": ParameterSpec(None, "channel", label="Optional x signal", allow_none=True, help="Convenience single pair. Leave null to use event-signal-to-other-channel pairs."),
            "y_signal": ParameterSpec(None, "channel", label="Optional y signal", allow_none=True, help="Convenience single pair. Leave null to use event-signal-to-other-channel pairs."),
            "detector": ParameterSpec("template", "select", label="Peak detector", options=["template", "peaks"]),
            "detector_params": _detector_params("template"),
            "window": _window(None),
            "t_before_s": ParameterSpec(20.0, "float", label="Before event / s", minimum=0, step=1.0),
            "t_after_s": ParameterSpec(20.0, "float", label="After event / s", minimum=0, step=1.0),
            "target_fs": ParameterSpec(None, "float", label="Target sampling / Hz", allow_none=True, minimum=0, step=1.0),
            "dt": ParameterSpec(None, "float", label="Output dt / s", allow_none=True, minimum=0, step=0.001),
            "baseline_window_s": ParameterSpec(None, "json", label="Baseline window / s", allow_none=True, help="Optional [start, stop] relative-time window to subtract from each epoch, e.g. [-10, -2]."),
            "max_lag_s": ParameterSpec(5.0, "float", label="Maximum lag / s", minimum=0, step=1.0),
            "detrend": ParameterSpec(True, "bool", label="Detrend before correlation"),
            "normalise": ParameterSpec(True, "bool", label="Normalise correlation"),
            "run_granger": ParameterSpec(False, "bool", label="Run Granger"),
            "granger_mode": ParameterSpec("mean_epoch", "select", label="Granger mode", options=["mean_epoch", "per_event"], help="mean_epoch is much faster; per_event tests each aligned epoch separately."),
            "granger_target_dt": ParameterSpec(1.0, "float", label="Granger target dt / s", allow_none=True, minimum=0, step=0.1),
            "granger_max_lag_steps": ParameterSpec(10, "int", label="Granger maximum lag steps", allow_none=True, minimum=1, step=1),
            "granger_max_samples": ParameterSpec(1000, "int", label="Granger maximum samples", allow_none=True, minimum=100, step=100),
            "granger_max_events": ParameterSpec(50, "int", label="Granger maximum events", allow_none=True, minimum=1, step=1),
        },
        replace=replace,
    )


register_defaults()


@dataclass(frozen=True, slots=True)
class StageConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageConfig":
        return cls(
            name=str(data["name"]),
            params=dict(data.get("params", {})),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisConfig":
        return cls(
            name=str(data["name"]),
            params=dict(data.get("params", {})),
            enabled=bool(data.get("enabled", True)),
        )


def create_stage(config: StageConfig | Mapping[str, Any]) -> Any:
    cfg = config if isinstance(config, StageConfig) else StageConfig.from_dict(config)
    return STAGE_REGISTRY.create(cfg.name, cfg.params)


def create_analysis(config: AnalysisConfig | Mapping[str, Any]) -> Any:
    cfg = config if isinstance(config, AnalysisConfig) else AnalysisConfig.from_dict(config)
    return ANALYSIS_REGISTRY.create(cfg.name, cfg.params)


def default_stage_params(name: str) -> dict[str, Any]:
    return STAGE_REGISTRY.default_params(name)


def default_analysis_params(name: str) -> dict[str, Any]:
    return ANALYSIS_REGISTRY.default_params(name)

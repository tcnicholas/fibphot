from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .registry import ParameterSpec, RegisteredObject


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _parse_json_text(text: str) -> Any:
    raw = (text or "").strip()
    if raw == "":
        return None
    return json.loads(raw)


def _is_iso_like_channel(name: str) -> bool:
    lo = str(name).strip().lower()
    return (
        "iso" in lo
        or "isos" in lo
        or "isob" in lo
        or lo in {"405", "405nm", "405_nm"}
        or "405" in lo
    )


def _non_iso_channels(channels: Iterable[str]) -> list[str]:
    names = [str(c) for c in channels]
    preferred = [c for c in names if not _is_iso_like_channel(c)]
    return preferred or names


def _best_iso_channel(channels: Iterable[str]) -> str | None:
    names = [str(c) for c in channels]
    lowered = [(n, n.lower()) for n in names]
    for n, lo in lowered:
        if lo == "iso" or lo.startswith("iso_") or lo.endswith("_iso"):
            return n
    for n, lo in lowered:
        if "iso" in lo or "isos" in lo or "isob" in lo:
            return n
    for n, lo in lowered:
        if "405" in lo:
            return n
    return None


def _labels_for_channels(
    channels: Iterable[str],
    default: Any,
    *,
    allow_none: bool = False,
) -> list[Any]:
    values: list[Any] = []
    if allow_none:
        values.append(None)
    for c in channels:
        if c not in values:
            values.append(c)
    if default is not None and default not in values:
        values.insert(0 if not allow_none else 1, default)
    return values or ([None] if allow_none else [default])


def _looks_like_control_parameter(name: str, label: str | None = None) -> bool:
    text = f"{name} {label or ''}".lower()
    return "control" in text or "isosbestic" in text or "isobestic" in text


def _looks_like_optional_pair_signal(name: str) -> bool:
    # Optional convenience fields such as x_signal/y_signal should not be
    # auto-filled.  Leaving them empty lets the analysis infer sensible pairs.
    return name.lower() in {"x_signal", "y_signal"}


class ParameterEditor:
    """Panel-backed editor generated from a RegisteredObject schema."""

    def __init__(self, pn: Any, *, title: str = "Parameters") -> None:
        self.pn = pn
        self.title = title
        self.spec: RegisteredObject | None = None
        self.widgets: dict[str, Any] = {}
        self.param_specs: dict[str, ParameterSpec] = {}
        self.channel_names: list[str] = []
        self.container = pn.Column(sizing_mode="stretch_width")

    def rebuild(
        self,
        spec: RegisteredObject | None,
        *,
        values: dict[str, Any] | None = None,
        channel_names: Iterable[str] = (),
    ) -> None:
        self.spec = spec
        self.channel_names = list(channel_names)
        self.widgets = {}
        self.param_specs = {}
        provided_values = dict(values or {})
        values = provided_values

        objects: list[Any] = []
        if spec is None:
            objects.append(self.pn.pane.Markdown("No object selected."))
            self.container.objects = objects
            return

        if spec.description:
            objects.append(
                self.pn.pane.Markdown(spec.description, margin=(0, 0, 8, 0))
            )

        for name, pspec in spec.parameters.items():
            provided = name in provided_values
            value = values.get(name, pspec.default)
            value = self._normalise_channel_value(
                name,
                pspec,
                value,
                provided=provided,
            )
            widget = self._make_widget(name, pspec, value)
            self.widgets[name] = widget
            self.param_specs[name] = pspec
            objects.append(widget)
            if pspec.help:
                objects.append(
                    self.pn.pane.Markdown(
                        f"<small>{pspec.help}</small>", margin=(-8, 0, 8, 4)
                    )
                )

        if not spec.parameters:
            objects.append(
                self.pn.pane.Markdown(
                    "This object has no registered parameters. Register a "
                    "parameter schema to get editable controls."
                )
            )
        self.container.objects = objects
        self._wire_dependencies(provided_values=provided_values)

    def _wire_dependencies(self, *, provided_values: dict[str, Any]) -> None:
        """Wire simple dependency rules between widgets.

        This is intentionally generic: a ParameterSpec can say that its value
        depends on another widget and provide defaults for each source value.
        The GUI then updates the dependent value whenever the source changes.
        It is mainly used for detector_params, which must switch between the
        template-matched detector schema and the local-peak detector schema.
        """
        for target_name, target_spec in self.param_specs.items():
            source_name = target_spec.depends_on
            defaults = target_spec.defaults_by_value
            if (
                not source_name
                or not defaults
                or source_name not in self.widgets
            ):
                continue

            source_widget = self.widgets[source_name]
            target_widget = self.widgets[target_name]

            def _callback(
                event: Any,
                *,
                tw=target_widget,
                ts=target_spec,
                mapping=defaults,
            ) -> None:
                if event.new in mapping:
                    self._set_widget_value(tw, ts, mapping[event.new])

            source_widget.param.watch(_callback, "value")

            # If the user provided detector='peaks' but omitted detector_params
            # in a saved config, use the detector-specific default instead of
            # the target field's static default. Do not overwrite explicit
            # target values on initial rebuild.
            if target_name not in provided_values:
                source_value = getattr(source_widget, "value", None)
                if source_value in defaults:
                    self._set_widget_value(
                        target_widget, target_spec, defaults[source_value]
                    )

    def _widget_value_for_param(self, spec: ParameterSpec, value: Any) -> Any:
        kind = spec.kind
        if kind == "channels":
            if self.channel_names:
                if value is None:
                    return []
                if isinstance(value, str):
                    if value.lower() == "all":
                        return _non_iso_channels(self.channel_names)
                    return [value] if value in self.channel_names else []
                if isinstance(value, list | tuple):
                    return [
                        str(v) for v in value if str(v) in self.channel_names
                    ]
                return []
            if isinstance(value, list | tuple):
                return ", ".join(str(v) for v in value)
            if value is None:
                return ""
            return str(value)
        if kind in {"dict", "window", "json"}:
            return _json_text(value)
        if kind == "callable":
            return "" if value is None else str(value)
        return value

    def _normalise_channel_value(
        self,
        name: str,
        spec: ParameterSpec,
        value: Any,
        *,
        provided: bool,
    ) -> Any:
        """Choose safe channel defaults once channel names are available.

        Single-channel fields become dropdowns populated from the loaded
        recording.  Multi-channel fields become multi-selects that default to
        all non-isosbestic channels.  Explicit saved values are preserved when
        still valid.
        """
        if not self.channel_names or spec.kind not in {"channel", "channels"}:
            return value

        names = list(self.channel_names)
        non_iso = _non_iso_channels(names)

        if spec.kind == "channels":
            if isinstance(value, str):
                if value.lower() == "all":
                    return non_iso
                if value in names:
                    return [value]
                parts = [p.strip() for p in value.split(",") if p.strip()]
                selected = [p for p in parts if p in names]
                if selected:
                    return selected
            if isinstance(value, list | tuple):
                selected = [str(v) for v in value if str(v) in names]
                if selected:
                    return selected
            if provided and value in (None, ""):
                return []
            return non_iso

        # Single-channel fields.
        if value in names:
            return value

        if _looks_like_control_parameter(name, spec.label):
            iso = _best_iso_channel(names)
            if iso is not None:
                return iso
            return (
                None
                if spec.allow_none
                else (non_iso[0] if non_iso else names[0])
            )

        if (
            spec.allow_none
            and value is None
            and _looks_like_optional_pair_signal(name)
        ):
            return None

        # Prefer biologically common non-isosbestic channels when the registered
        # default is absent from this recording.
        preferences = ("gcamp", "green", "465", "rgeco", "red", "560")
        lowered = [(n, n.lower()) for n in non_iso]
        for token in preferences:
            for n, lo in lowered:
                if token in lo:
                    return n
        return non_iso[0] if non_iso else (names[0] if names else value)

    def _set_widget_value(
        self, widget: Any, spec: ParameterSpec, value: Any
    ) -> None:
        widget.value = self._widget_value_for_param(spec, value)

    def update_channels(self, channel_names: Iterable[str]) -> None:
        if list(channel_names) == self.channel_names:
            return
        current = self.values(safe=True)
        self.rebuild(self.spec, values=current, channel_names=channel_names)

    def values(self, *, safe: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, widget in self.widgets.items():
            pspec = self.param_specs[name]
            try:
                out[name] = self._value_from_widget(widget, pspec)
            except Exception:
                if not safe:
                    raise
                out[name] = pspec.default
        return out

    def _make_widget(self, name: str, spec: ParameterSpec, value: Any) -> Any:
        label = spec.label or name
        kind = spec.kind
        pnw = self.pn.widgets

        if kind == "bool":
            return pnw.Checkbox(name=label, value=bool(value))

        if kind == "int" and not (spec.allow_none and value is None):
            return pnw.IntInput(
                name=label,
                value=int(value),
                start=spec.minimum,
                end=spec.maximum,
                step=int(spec.step or 1),
            )

        if kind == "float" and not (spec.allow_none and value is None):
            return pnw.FloatInput(
                name=label,
                value=float(value),
                start=spec.minimum,
                end=spec.maximum,
                step=float(spec.step or 0.1),
            )

        if kind == "select":
            options = spec.options or []
            return pnw.Select(name=label, value=value, options=options)

        if kind == "channel":
            options = _labels_for_channels(
                self.channel_names,
                value,
                allow_none=spec.allow_none,
            )
            return pnw.Select(name=label, value=value, options=options)

        if kind == "channels":
            if self.channel_names:
                selected = self._widget_value_for_param(spec, value)
                return pnw.MultiChoice(
                    name=label,
                    value=list(selected or []),
                    options=list(self.channel_names),
                    sizing_mode="stretch_width",
                )
            if isinstance(value, list | tuple):
                text = ", ".join(str(v) for v in value)
            elif value is None:
                text = ""
            else:
                text = str(value)
            return pnw.TextInput(name=label, value=text)

        if kind in {"dict", "window"}:
            return pnw.TextAreaInput(
                name=label, value=_json_text(value), height=120
            )

        if kind in {"json", "callable"} or (spec.allow_none and value is None):
            if kind == "callable":
                return pnw.TextInput(
                    name=label, value="" if value is None else str(value)
                )
            return pnw.TextAreaInput(
                name=label, value=_json_text(value), height=90
            )

        return pnw.TextInput(
            name=label, value="" if value is None else str(value)
        )

    def _value_from_widget(self, widget: Any, spec: ParameterSpec) -> Any:
        value = widget.value
        kind = spec.kind

        if kind == "channels":
            if isinstance(value, list | tuple):
                selected = [str(v) for v in value if str(v)]
                return selected if selected else None
            text = str(value or "").strip()
            if text == "" or text.lower() in {"none", "null"}:
                return None
            if text.lower() == "all":
                return "all"
            parts = [p.strip() for p in text.split(",") if p.strip()]
            return parts if len(parts) > 1 else (parts[0] if parts else None)

        if kind == "int":
            if spec.allow_none and str(value).strip().lower() in {
                "",
                "none",
                "null",
            }:
                return None
            return int(value)

        if kind == "float":
            if spec.allow_none and str(value).strip().lower() in {
                "",
                "none",
                "null",
            }:
                return None
            return float(value)

        if kind == "bool":
            return bool(value)

        if kind in {"dict", "window", "json"}:
            return _parse_json_text(str(value))

        if kind == "callable":
            text = str(value or "").strip()
            return text or None

        return value

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .registry import ANALYSIS_REGISTRY, STAGE_REGISTRY, AnalysisConfig, StageConfig
from .session import GuiSession
from .widgets import ParameterEditor


APP_CSS = """
:root {
  --fibphot-bg: #f5f7fb;
  --fibphot-surface: #ffffff;
  --fibphot-surface-soft: #f8fafc;
  --fibphot-border: #dbe3ef;
  --fibphot-text: #182433;
  --fibphot-muted: #5e6b7a;
  --fibphot-primary: #2564a0;
  --fibphot-primary-dark: #153f67;
  --fibphot-accent: #15a3a3;
}
html, body, .bk-root {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--fibphot-text);
  background: var(--fibphot-bg);
}
.fibphot-app {
  background: radial-gradient(circle at top left, #eef7ff 0, #f5f7fb 33%, #ffffff 100%);
}
.fibphot-header {
  align-items: center;
  padding: 0.65rem 1rem;
  border-bottom: 1px solid var(--fibphot-border);
  background: linear-gradient(135deg, #102f4e 0%, #2564a0 55%, #15a3a3 120%);
  color: white;
  box-shadow: 0 8px 24px rgba(16, 47, 78, 0.18);
}
.fibphot-title h1,
.fibphot-title h2,
.fibphot-title h3,
.fibphot-title p {
  color: white;
  margin: 0;
}
.fibphot-title h2 {
  font-weight: 650;
  letter-spacing: -0.03em;
}
.fibphot-title p {
  margin-top: 0.12rem;
  font-size: 0.82rem;
  opacity: 0.82;
}
.fibphot-sidebar {
  overflow: auto;
  min-width: 320px;
  max-width: 1000px;
  height: calc(100vh - 72px);
  padding: 0.75rem 0.9rem 0.75rem 0.75rem;
  border-right: 1px solid var(--fibphot-border);
  background: rgba(248, 250, 252, 0.92);
  backdrop-filter: blur(8px);
}
.fibphot-main {
  overflow: auto;
  height: calc(100vh - 72px);
  padding: 0.75rem;
}
.fibphot-card {
  border: 1px solid var(--fibphot-border);
  border-radius: 14px;
  background: var(--fibphot-surface);
  box-shadow: 0 8px 22px rgba(24, 36, 51, 0.055);
  overflow: hidden;
}
.fibphot-card .card-header,
.fibphot-card .accordion-header,
.fibphot-card .bk-Card-header {
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border-bottom: 1px solid var(--fibphot-border);
  color: var(--fibphot-primary-dark);
  font-weight: 650;
}
.fibphot-resizable-y {
  resize: vertical;
  overflow: auto;
  min-height: 210px;
  max-height: 85vh;
}
.fibphot-muted,
.fibphot-muted p {
  color: rgba(255, 255, 255, 0.78);
  font-size: 0.84rem;
}
.bk-input, input, textarea, select {
  border-radius: 8px !important;
}
.bk-btn {
  border-radius: 9px !important;
  font-weight: 600 !important;
}
"""

PLOT_PRIMARY = "#2564A0"
PLOT_RAW = "#8A97A8"
PLOT_PEAKS = "#D54B5B"
PLOT_MEAN = "#0F766E"
PLOT_SHADE = "#15A3A3"

# Professional, high-contrast channel palette.  Semantic channel names are
# mapped onto intuitive colours before falling back to this palette.
PLOT_GREEN = "#059669"
PLOT_RED = "#D9485F"
PLOT_BLUE = "#2564A0"
PLOT_PURPLE = "#7C3AED"
PLOT_AMBER = "#D97706"
PLOT_TEAL = "#0D9488"
PLOT_SLATE = "#64748B"
PLOT_CHANNELS = (
    PLOT_BLUE,
    PLOT_RED,
    PLOT_GREEN,
    PLOT_PURPLE,
    PLOT_AMBER,
    PLOT_TEAL,
    "#BE123C",
    "#4D7C0F",
)

def _require_panel():
    try:
        import panel as pn  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The fibphot GUI requires Panel and Bokeh. Install with `pip install panel bokeh pandas` "
            "or `pip install -e .[gui]` from the package root."
        ) from exc
    return pn


def _downsample(
    t: np.ndarray,
    y: np.ndarray,
    max_points: int,
    *,
    preserve_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a trace while optionally preserving important time points.

    Simple decimation can make event markers appear vertically offset from the
    plotted trace, because sharp peak tops may be omitted from the line while
    the event marker is drawn at the full-resolution peak value.  When
    ``preserve_times`` is supplied, the nearest full-resolution samples are
    forced into the plotted trace so markers and line agree visually.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size <= max_points:
        return t, y

    step = int(np.ceil(t.size / max_points))
    idx = np.arange(0, t.size, step, dtype=int)

    if preserve_times is not None:
        preserve = np.asarray(preserve_times, dtype=float)
        preserve = preserve[np.isfinite(preserve)]
        if preserve.size:
            keep = (preserve >= t[0]) & (preserve <= t[-1])
            preserve = preserve[keep]
        if preserve.size:
            peak_idx = np.searchsorted(t, preserve)
            peak_idx = np.clip(peak_idx, 0, t.size - 1)
            prev_idx = np.clip(peak_idx - 1, 0, t.size - 1)
            use_prev = np.abs(t[prev_idx] - preserve) < np.abs(t[peak_idx] - preserve)
            peak_idx[use_prev] = prev_idx[use_prev]
            idx = np.unique(np.concatenate([idx, peak_idx]))

    return t[idx], y[idx]


def _values_at_times(
    t: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Return nearest trace values at event times."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    times = np.asarray(times, dtype=float)
    if t.size == 0 or times.size == 0:
        return np.asarray([], dtype=float)
    idx = np.searchsorted(t, times)
    idx = np.clip(idx, 0, t.size - 1)
    prev_idx = np.clip(idx - 1, 0, t.size - 1)
    use_prev = np.abs(t[prev_idx] - times) < np.abs(t[idx] - times)
    idx[use_prev] = prev_idx[use_prev]
    return y[idx]


def _first_value(options: dict[str, str]) -> str | None:
    return next(iter(options.values()), None)


def _state_option_label(index: int, state: Any) -> str:
    subject = getattr(state, "subject", None) or "unknown"
    meta = getattr(state, "metadata", {}) or {}
    parent = meta.get("source_parent") or Path(str(meta.get("source_path", ""))).parent.name
    return f"{index}: {subject} / {parent}"


def _parse_option_index(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).split(":", 1)[0])
    except Exception:
        return None


def _split_groupby(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def _channel_colour(channel: str | None, *, fallback_index: int = 0) -> str:
    """Return a modern semantic colour for a photometry channel name."""
    if channel is None:
        return PLOT_CHANNELS[fallback_index % len(PLOT_CHANNELS)]
    lo = str(channel).strip().lower().replace("-", "_")

    # Isosbestic/control channels: keep visually muted.
    if "iso" in lo or "isos" in lo or "isob" in lo or "405" in lo:
        return PLOT_SLATE

    # Common colour/channel conventions.
    if (
        "green" in lo
        or "gcamp" in lo
        or "egfp" in lo
        or "gfp" in lo
        or "465" in lo
        or "470" in lo
    ):
        return PLOT_GREEN
    if (
        "red" in lo
        or "rgeco" in lo
        or "jrcamp" in lo
        or "rcamp" in lo
        or "tdtomato" in lo
        or "tomato" in lo
        or "mcherry" in lo
        or "560" in lo
        or "565" in lo
        or "590" in lo
    ):
        return PLOT_RED
    if "blue" in lo or "dapi" in lo or "cyan" in lo:
        return PLOT_BLUE
    if "purple" in lo or "violet" in lo or "uv" in lo:
        return PLOT_PURPLE
    if "yellow" in lo or "amber" in lo or "yfp" in lo:
        return PLOT_AMBER
    if "teal" in lo or "aqua" in lo:
        return PLOT_TEAL

    return PLOT_CHANNELS[fallback_index % len(PLOT_CHANNELS)]


def _channel_raw_colour(channel: str | None, *, fallback_index: int = 0) -> str:
    """Muted colour for raw trace overlays of a given channel."""
    if channel is None:
        return PLOT_RAW
    if "iso" in str(channel).lower() or "405" in str(channel).lower():
        return "#94A3B8"
    return _channel_colour(channel, fallback_index=fallback_index)


class FibPhotGUI:
    """Interactive local GUI for fibphot.

    The GUI is registry-driven. Adding a new stage or analysis requires
    registering it in ``fibphot.gui.registry`` or an extension module; the app
    itself does not need to know the concrete class.
    """

    def __init__(self):
        self.pn = _require_panel()
        self.pn.extension("tabulator", sizing_mode="stretch_width")
        if APP_CSS not in self.pn.config.raw_css:
            self.pn.config.raw_css.append(APP_CSS)
        self.session = GuiSession()

        # File/session widgets ------------------------------------------------
        self.path_input = self.pn.widgets.TextInput(
            name="Input file",
            placeholder="/path/to/session.doric or processed_state.h5",
            sizing_mode="stretch_width",
        )
        self.load_button = self.pn.widgets.Button(name="Load", button_type="primary")
        self.browse_input = self.pn.widgets.Checkbox(name="Browse files", value=False)
        try:
            self.input_file_selector = self.pn.widgets.FileSelector(
                name="Browse input file",
                directory=str(Path.cwd()),
                file_pattern="*",
                only_files=True,
                visible=False,
                sizing_mode="stretch_width",
            )
        except Exception:  # pragma: no cover - older Panel versions.
            self.input_file_selector = self.pn.pane.Markdown(
                "File browser unavailable in this Panel version.",
                visible=False,
            )
        self.session_file_path = self.pn.widgets.TextInput(
            name="fibphot session file",
            value="fibphot_session.h5",
            placeholder="Relative to output directory, or absolute path",
            sizing_mode="stretch_width",
        )
        self.save_session_button = self.pn.widgets.Button(name="Save GUI session")
        self.load_session_button = self.pn.widgets.Button(name="Load GUI session", button_type="primary")
        self.status = self.pn.pane.Markdown("No file loaded.")
        self.history_policy = self.pn.widgets.Select(
            name="History policy",
            value="raw",
            options=["none", "raw", "all", "checkpoints"],
        )
        self.output_dir_input = self.pn.widgets.TextInput(
            name="Output directory",
            value=str(Path.cwd()),
            placeholder="Directory for exports, pipelines and result files",
            sizing_mode="stretch_width",
        )
        self.output_dir_button = self.pn.widgets.Button(
            name="Apply output directory",
            button_type="primary",
        )

        # Plot widgets --------------------------------------------------------
        self.channel_select = self.pn.widgets.MultiChoice(
            name="Channels",
            options=[],
            value=[],
            placeholder="Choose one or more channels to plot",
            sizing_mode="stretch_width",
        )
        self.show_raw = self.pn.widgets.Checkbox(name="Overlay raw", value=True)
        self.show_processed = self.pn.widgets.Checkbox(name="Show processed", value=True)
        self.show_peaks = self.pn.widgets.Checkbox(name="Overlay latest peak result", value=True)
        self.max_points = self.pn.widgets.IntInput(
            name="Max plotted points",
            value=20000,
            start=1000,
            end=500000,
            step=1000,
        )
        self.plot_backend = self.pn.widgets.Select(
            name="Trace rendering",
            value="svg",
            options={"SVG (crisp/vector)": "svg", "Canvas (faster)": "canvas"},
        )
        self.trace_scope = self.pn.widgets.Select(
            name="Trace content",
            value="single",
            options={
                "Current session trace": "single",
                "Batch sessions / average": "batch",
                "Selected analysis result": "analysis",
            },
        )
        self.batch_trace_mode = self.pn.widgets.Select(
            name="Batch display",
            value="both",
            options={
                "Overlay selected sessions": "overlay",
                "Average only": "mean",
                "Overlay + average": "both",
            },
        )
        self.batch_trace_error = self.pn.widgets.Select(
            name="Batch average error",
            value="sem",
            options={"SEM": "sem", "SD": "std", "None": "none"},
        )
        self.batch_trace_sessions = self.pn.widgets.MultiChoice(
            name="Batch trace sessions",
            options=[],
            value=[],
            placeholder="Blank = all sessions; or choose specific sessions",
            sizing_mode="stretch_width",
        )
        self.batch_trace_groupby = self.pn.widgets.TextInput(
            name="Batch trace group by",
            value="",
            placeholder="Optional, e.g. genotype or subject,day",
            sizing_mode="stretch_width",
        )
        self.batch_trace_group = self.pn.widgets.Select(
            name="Batch trace group",
            options={"All groups": ""},
            value="",
        )
        self.result_plot_mode = self.pn.widgets.Select(
            name="Analysis plot",
            value="auto",
            options={
                "Auto": "auto",
                "Session overlay": "trace",
                "Aligned epochs": "epochs",
                "Connectivity / lag curve": "curve",
            },
        )
        self.epoch_display = self.pn.widgets.Select(
            name="Epoch display",
            value="both",
            options={
                "Individual epochs": "individual",
                "Average only": "mean",
                "Individual + average": "both",
            },
        )
        self.epoch_error = self.pn.widgets.Select(
            name="Epoch uncertainty",
            value="sem",
            options={"SEM": "sem", "SD": "std", "IQR": "iqr", "None": "none"},
        )
        self.aligned_channel_mode = self.pn.widgets.Select(
            name="Aligned channels",
            value="all",
            options={
                "All aligned channels": "all",
                "Selected trace channel(s)": "selected",
            },
        )
        self.plot_pane = self.pn.pane.Bokeh(
            min_height=470,
            sizing_mode="stretch_both",
        )

        # Pipeline widgets ----------------------------------------------------
        stage_options = STAGE_REGISTRY.options()
        first_stage = _first_value(stage_options)
        self.stage_select = self.pn.widgets.Select(
            name="Add processing stage",
            options=stage_options,
            value=first_stage,
        )
        self.stage_add_editor = ParameterEditor(self.pn, title="Stage parameters")
        self.stage_add_editor.rebuild(
            STAGE_REGISTRY.get(first_stage) if first_stage else None,
            values=self._smart_stage_values(first_stage, STAGE_REGISTRY.default_params(first_stage)) if first_stage else None,
            channel_names=self._channel_names(),
        )
        self.add_stage_button = self.pn.widgets.Button(name="Add stage", button_type="success")

        self.pipeline_table = self.pn.widgets.Tabulator(
            value=self._pipeline_frame(),
            height=230,
            selectable=1,
            disabled=True,
        )
        self.edit_stage_select = self.pn.widgets.Select(name="Selected stage", options=[])
        self.edit_stage_enabled = self.pn.widgets.Checkbox(name="Enabled", value=True)
        self.stage_edit_editor = ParameterEditor(self.pn, title="Selected stage parameters")
        self.apply_stage_button = self.pn.widgets.Button(name="Apply stage edit", button_type="primary")
        self.remove_stage_button = self.pn.widgets.Button(name="Remove stage", button_type="danger")
        self.stage_up_button = self.pn.widgets.Button(name="Move up")
        self.stage_down_button = self.pn.widgets.Button(name="Move down")
        self.undo_button = self.pn.widgets.Button(name="Undo")
        self.redo_button = self.pn.widgets.Button(name="Redo")
        self.recompute_button = self.pn.widgets.Button(name="Recompute", button_type="primary")
        self.pipeline_json_path = self.pn.widgets.TextInput(name="Pipeline JSON", value="pipeline.json")
        self.save_pipeline_button = self.pn.widgets.Button(name="Save pipeline")
        self.load_pipeline_button = self.pn.widgets.Button(name="Load pipeline")

        # Analysis widgets ----------------------------------------------------
        analysis_options = ANALYSIS_REGISTRY.options()
        first_analysis = _first_value(analysis_options)
        self.analysis_select = self.pn.widgets.Select(
            name="Analysis",
            options=analysis_options,
            value=first_analysis,
        )
        self.analysis_editor = ParameterEditor(self.pn, title="Analysis parameters")
        self.analysis_editor.rebuild(
            ANALYSIS_REGISTRY.get(first_analysis) if first_analysis else None,
            channel_names=self._channel_names(),
        )
        self.run_analysis_button = self.pn.widgets.Button(name="Run analysis", button_type="success")
        self.rerun_analyses_button = self.pn.widgets.Button(name="Rerun saved analyses", button_type="primary")
        self.clear_results_button = self.pn.widgets.Button(name="Clear results")
        self.results_summary = self.pn.pane.Markdown("No analyses run.")
        self.result_select = self.pn.widgets.Select(name="Selected result", options=[])
        self.metrics_table = self.pn.widgets.Tabulator(value=self._empty_frame(), height=210, disabled=True)
        self.arrays_table = self.pn.widgets.Tabulator(value=self._empty_frame(), height=260, disabled=True)

        # Export widgets ------------------------------------------------------
        self.export_dir = self.pn.widgets.TextInput(
            name="Export folder",
            value="fibphot_export",
            placeholder="Relative to output directory, or an absolute path",
        )
        self.export_button = self.pn.widgets.Button(name="Export all", button_type="primary")
        self.export_selected_button = self.pn.widgets.Button(name="Export selected result")
        self.export_status = self.pn.pane.Markdown("")

        # Batch/multi-session widgets ---------------------------------------
        self.batch_dir_input = self.pn.widgets.TextInput(
            name="Batch directory",
            value="",
            placeholder="Folder containing .doric files",
            sizing_mode="stretch_width",
        )
        self.batch_pattern = self.pn.widgets.TextInput(
            name="File pattern",
            value="**/*.doric",
            sizing_mode="stretch_width",
        )
        self.batch_metadata_path = self.pn.widgets.TextInput(
            name="Metadata CSV/Excel",
            value="",
            placeholder="Optional metadata table",
            sizing_mode="stretch_width",
        )
        self.batch_metadata_on = self.pn.widgets.TextInput(
            name="Metadata match column",
            value="subject",
        )
        self.load_batch_button = self.pn.widgets.Button(name="Load batch", button_type="primary")
        self.process_batch_button = self.pn.widgets.Button(name="Apply pipeline to batch", button_type="primary")
        self.run_batch_selected_button = self.pn.widgets.Button(name="Run selected analysis on batch", button_type="success")
        self.rerun_batch_button = self.pn.widgets.Button(name="Rerun saved analyses on batch", button_type="primary")
        self.export_batch_button = self.pn.widgets.Button(name="Export batch", button_type="primary")
        self.batch_status = self.pn.pane.Markdown("No batch loaded.")
        self.batch_session_select = self.pn.widgets.Select(name="Inspect session", options=[])
        self.open_batch_session_button = self.pn.widgets.Button(name="Open selected session")
        self.batch_groupby = self.pn.widgets.TextInput(
            name="Group by metadata",
            value="",
            placeholder="e.g. genotype, day or subject,day",
            sizing_mode="stretch_width",
        )
        self.batch_values = self.pn.widgets.TextInput(
            name="Metrics to summarise",
            value="",
            placeholder="Blank = all numeric metrics; or comma-separated names",
            sizing_mode="stretch_width",
        )
        self.summarise_batch_button = self.pn.widgets.Button(name="Summarise batch metrics")
        self.batch_sessions_table = self.pn.widgets.Tabulator(value=self._empty_frame(), height=240, disabled=True)
        self.batch_metrics_table = self.pn.widgets.Tabulator(value=self._empty_frame(), height=260, disabled=True)
        self.batch_grouped_table = self.pn.widgets.Tabulator(value=self._empty_frame(), height=260, disabled=True)
        self.batch_plot_pane = self.pn.pane.Bokeh(height=320, sizing_mode="stretch_width")

        # Layout widgets ------------------------------------------------------
        self.sidebar_width = self.pn.widgets.IntSlider(
            name="Sidebar width",
            value=540,
            start=320,
            end=1000,
            step=10,
            width=260,
        )
        self._sidebar_container = None

        self._wire_callbacks()
        self._refresh_all()

    # ------------------------------------------------------------------
    # Small data helpers
    # ------------------------------------------------------------------
    def _empty_frame(self):
        import pandas as pd

        return pd.DataFrame()

    def _channel_names(self) -> list[str]:
        if self.session.current_state is None:
            if self.session.current_collection is not None and len(self.session.current_collection):
                return list(self.session.current_collection[0].channel_names)
            return []
        return list(self.session.current_state.channel_names)

    def _infer_control_channel(self, names: list[str]) -> str | None:
        if not names:
            return None
        # Prefer exact-ish isosbestic names, then closest names containing iso.
        lowered = [(n, str(n).lower()) for n in names]
        for n, lo in lowered:
            if lo == "iso" or lo.startswith("iso_") or lo.endswith("_iso"):
                return n
        for n, lo in lowered:
            if "iso" in lo or "isos" in lo or "isob" in lo:
                return n
        return None

    def _is_isosbestic_channel(self, name: str | None) -> bool:
        if name is None:
            return False
        lo = str(name).lower()
        return (
            lo == "iso"
            or lo.startswith("iso_")
            or lo.endswith("_iso")
            or "isob" in lo
            or "isos" in lo
            or "405" in lo
        )

    def _default_plot_channels(self, names: list[str]) -> list[str]:
        if not names:
            return []
        non_iso = [n for n in names if not self._is_isosbestic_channel(n)]
        return non_iso or names[:1]

    def _resolve_channel_from_names(
        self,
        channel: str | None,
        names: list[str] | tuple[str, ...],
    ) -> str | None:
        if channel is None:
            return None
        requested = str(channel)
        if requested in names:
            return requested
        requested_lower = requested.lower()
        for name in names:
            if str(name).lower() == requested_lower:
                return str(name)
        return None

    def _selected_plot_channels(self) -> list[str]:
        names = self._channel_names()
        value = self.channel_select.value
        if isinstance(value, (list, tuple, set)):
            requested = [str(v) for v in value if v not in (None, "")]
        elif value in (None, ""):
            requested = []
        else:
            requested = [str(value)]

        resolved: list[str] = []
        for item in requested:
            name = self._resolve_channel_from_names(item, names)
            if name is not None and name not in resolved:
                resolved.append(name)
        return resolved

    def _primary_plot_channel(self) -> str | None:
        channels = self._selected_plot_channels()
        if channels:
            return channels[0]
        names = self._channel_names()
        defaults = self._default_plot_channels(names)
        return defaults[0] if defaults else None

    def _smart_stage_values(self, key: str, values: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(values or {})
        if key not in {"IsosbesticRegression", "IsosbesticDff"}:
            return out
        names = self._channel_names()
        control = out.get("control")
        if control not in names:
            control = self._infer_control_channel(names)
            out["control"] = control
        channels = out.get("channels")
        if control is None:
            # Force an explicit user choice rather than pre-filling a bad value
            # that will fail at runtime.
            out["channels"] = None
            return out
        if channels in (None, "", "all") or channels == ["rgeco", "gcamp"]:
            out["channels"] = [n for n in names if n != control]
        return out

    def _pipeline_frame(self):
        import pandas as pd

        rows = []
        for i, cfg in enumerate(self.session.pipeline):
            spec = STAGE_REGISTRY.get(cfg.name) if cfg.name in STAGE_REGISTRY else None
            rows.append(
                {
                    "#": i,
                    "enabled": cfg.enabled,
                    "stage": spec.display_name if spec is not None else cfg.name,
                    "key": cfg.name,
                    "params": json.dumps(cfg.params, ensure_ascii=False),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Callback wiring
    # ------------------------------------------------------------------
    def _wire_callbacks(self) -> None:
        self.load_button.on_click(self._on_load)
        self.browse_input.param.watch(self._on_browse_input_toggle, "value")
        if hasattr(self.input_file_selector, "param"):
            self.input_file_selector.param.watch(self._on_input_file_selected, "value")
        self.save_session_button.on_click(self._on_save_session)
        self.load_session_button.on_click(self._on_load_session)
        self.output_dir_button.on_click(self._on_output_dir)
        self.history_policy.param.watch(self._on_history_policy, "value")
        for widget in (
            self.channel_select,
            self.show_raw,
            self.show_processed,
            self.show_peaks,
            self.max_points,
            self.plot_backend,
            self.trace_scope,
            self.batch_trace_mode,
            self.batch_trace_error,
            self.batch_trace_sessions,
            self.batch_trace_group,
            self.result_plot_mode,
            self.epoch_display,
            self.epoch_error,
            self.aligned_channel_mode,
        ):
            widget.param.watch(lambda *_: self._update_plot(), "value")
        self.batch_trace_groupby.param.watch(lambda *_: (self._refresh_batch_trace_groups(), self._update_plot()), "value")
        self.sidebar_width.param.watch(self._on_sidebar_width, "value")

        self.stage_select.param.watch(self._on_stage_choice, "value")
        self.add_stage_button.on_click(self._on_add_stage)
        self.edit_stage_select.param.watch(self._on_edit_stage_choice, "value")
        self.apply_stage_button.on_click(self._on_apply_stage_edit)
        self.remove_stage_button.on_click(self._on_remove_stage)
        self.stage_up_button.on_click(lambda *_: self._move_stage(-1))
        self.stage_down_button.on_click(lambda *_: self._move_stage(1))
        self.undo_button.on_click(lambda *_: self._undo_redo("undo"))
        self.redo_button.on_click(lambda *_: self._undo_redo("redo"))
        self.recompute_button.on_click(self._on_recompute)
        self.save_pipeline_button.on_click(self._on_save_pipeline)
        self.load_pipeline_button.on_click(self._on_load_pipeline)

        self.analysis_select.param.watch(self._on_analysis_choice, "value")
        self.result_select.param.watch(lambda *_: self._update_plot(), "value")
        self.run_analysis_button.on_click(self._on_run_analysis)
        self.rerun_analyses_button.on_click(self._on_rerun_analyses)
        self.clear_results_button.on_click(self._on_clear_results)

        self.export_button.on_click(self._on_export)
        self.export_selected_button.on_click(self._on_export_selected)

        self.load_batch_button.on_click(self._on_load_batch)
        self.process_batch_button.on_click(self._on_process_batch)
        self.run_batch_selected_button.on_click(self._on_run_batch_selected_analysis)
        self.rerun_batch_button.on_click(self._on_rerun_batch_analyses)
        self.export_batch_button.on_click(self._on_export_batch)
        self.open_batch_session_button.on_click(self._on_open_batch_session)
        self.summarise_batch_button.on_click(self._on_summarise_batch)

    def _set_error(self, exc: Exception) -> None:
        self.status.object = f"**Error:** `{type(exc).__name__}: {exc}`"

    def _on_browse_input_toggle(self, *_: Any) -> None:
        if hasattr(self.input_file_selector, "visible"):
            self.input_file_selector.visible = bool(self.browse_input.value)

    def _on_input_file_selected(self, event: Any) -> None:
        value = event.new
        if isinstance(value, (list, tuple)) and value:
            self.path_input.value = str(value[0])
        elif isinstance(value, str) and value:
            self.path_input.value = value

    def _looks_like_gui_session(self, path: str | Path) -> bool:
        p = Path(path).expanduser()
        if not p.exists() or p.suffix.lower() not in {".h5", ".hdf5"}:
            return False
        try:
            import h5py

            with h5py.File(p, "r") as h5:
                return str(h5.attrs.get("schema", "")) == "fibphot_gui_session"
        except Exception:
            return False

    def _sync_widgets_from_session(self) -> None:
        self.output_dir_input.value = str(self.session.output_dir)
        if self.session.source_path:
            self.path_input.value = str(self.session.source_path)
        if self.session.batch_source_dir:
            self.batch_dir_input.value = str(self.session.batch_source_dir)

    def _on_save_session(self, *_: Any) -> None:
        try:
            path = self.session.save_session_h5(self.session_file_path.value)
            self.status.object = f"Saved GUI session to `{path}`."
        except Exception as exc:
            self._set_error(exc)

    def _on_load_session(self, *_: Any) -> None:
        try:
            # For convenience, if the main input path points to a GUI session and
            # the dedicated session field is still at its default, use it.
            candidate = self.session_file_path.value.strip()
            main = self.path_input.value.strip()
            if main and self._looks_like_gui_session(main) and candidate == "fibphot_session.h5":
                candidate = main
            path = self.session.load_session_h5(candidate)
            self._sync_widgets_from_session()
            self.status.object = f"Loaded GUI session `{path.name}` with {len(self.session.results)} result(s)."
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    # ------------------------------------------------------------------
    # Session and pipeline callbacks
    # ------------------------------------------------------------------
    def _on_load(self, *_: Any) -> None:
        try:
            path = self.path_input.value.strip()
            if not path:
                raise ValueError("Enter a .doric, .h5, .hdf5 or saved fibphot GUI session file path.")
            if self._looks_like_gui_session(path):
                loaded = self.session.load_session_h5(path)
                self.session_file_path.value = str(loaded)
                self._sync_widgets_from_session()
                self.status.object = f"Loaded GUI session `{loaded.name}` with {len(self.session.results)} result(s)."
                self._refresh_all()
                return
            st = self.session.load(path)
            self.status.object = (
                f"Loaded `{Path(path).name}` — {st.n_signals} channel(s), "
                f"{st.n_samples} sample(s), {st.sampling_rate:.3g} Hz."
            )
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    def _on_history_policy(self, *_: Any) -> None:
        self.session.history_policy = self.history_policy.value

    def _on_output_dir(self, *_: Any) -> None:
        try:
            out = self.session.set_output_dir(self.output_dir_input.value.strip() or Path.cwd())
            self.output_dir_input.value = str(out)
            self.status.object = f"Output directory set to `{out}`."
        except Exception as exc:
            self._set_error(exc)

    def _on_sidebar_width(self, *_: Any) -> None:
        sidebar = getattr(self, "_sidebar_container", None)
        if sidebar is None:
            return
        width = int(self.sidebar_width.value or 540)
        sidebar.width = width
        sidebar.styles = {**(sidebar.styles or {}), "width": f"{width}px"}

    def _on_stage_choice(self, event: Any) -> None:
        key = event.new
        self.stage_add_editor.rebuild(
            STAGE_REGISTRY.get(key),
            values=self._smart_stage_values(key, STAGE_REGISTRY.default_params(key)),
            channel_names=self._channel_names(),
        )

    def _on_add_stage(self, *_: Any) -> None:
        try:
            key = str(self.stage_select.value)
            cfg = StageConfig(key, self._smart_stage_values(key, self.stage_add_editor.values()), True)
            self.session.add_stage(cfg)
            self.status.object = f"Added `{STAGE_REGISTRY.get(key).display_name}` and recomputed."
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    def _selected_stage(self) -> int | None:
        value = self.edit_stage_select.value
        if value in (None, ""):
            return None
        return int(str(value).split(":", 1)[0])

    def _on_edit_stage_choice(self, *_: Any) -> None:
        idx = self._selected_stage()
        if idx is None or idx >= len(self.session.pipeline):
            self.stage_edit_editor.rebuild(None, channel_names=self._channel_names())
            return
        cfg = self.session.pipeline[idx]
        self.edit_stage_enabled.value = cfg.enabled
        self.stage_edit_editor.rebuild(
            STAGE_REGISTRY.get(cfg.name),
            values=cfg.params,
            channel_names=self._channel_names(),
        )

    def _on_apply_stage_edit(self, *_: Any) -> None:
        try:
            idx = self._selected_stage()
            if idx is None:
                raise ValueError("Select a stage to edit.")
            old = self.session.pipeline[idx]
            cfg = StageConfig(old.name, self._smart_stage_values(old.name, self.stage_edit_editor.values()), self.edit_stage_enabled.value)
            self.session.update_stage(idx, cfg)
            self.status.object = f"Updated stage {idx}: `{STAGE_REGISTRY.get(cfg.name).display_name}`."
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    def _on_remove_stage(self, *_: Any) -> None:
        try:
            idx = self._selected_stage()
            if idx is None:
                raise ValueError("Select a stage to remove.")
            name = self.session.pipeline[idx].name
            self.session.remove_stage(idx)
            label = STAGE_REGISTRY.get(name).display_name if name in STAGE_REGISTRY else name
            self.status.object = f"Removed stage {idx}: `{label}`."
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    def _move_stage(self, delta: int) -> None:
        try:
            idx = self._selected_stage()
            if idx is None:
                raise ValueError("Select a stage to move.")
            self.session.move_stage(idx, delta)
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    def _undo_redo(self, action: str) -> None:
        try:
            ok = self.session.undo() if action == "undo" else self.session.redo()
            self.status.object = f"{action.title()} {'applied' if ok else 'stack is empty'}."
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    def _on_recompute(self, *_: Any) -> None:
        try:
            self.session.recompute()
            self.status.object = "Recomputed current state."
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    def _on_save_pipeline(self, *_: Any) -> None:
        try:
            path = self.session.save_pipeline_json(self.pipeline_json_path.value)
            self.status.object = f"Saved pipeline to `{path}`."
        except Exception as exc:
            self._set_error(exc)

    def _on_load_pipeline(self, *_: Any) -> None:
        try:
            path = self.session.output_path(self.pipeline_json_path.value)
            self.session.load_pipeline_json(self.pipeline_json_path.value, recompute=self.session.loaded)
            self.status.object = f"Loaded pipeline from `{path}`."
            self._refresh_all()
        except Exception as exc:
            self._set_error(exc)

    # ------------------------------------------------------------------
    # Analysis callbacks
    # ------------------------------------------------------------------
    def _on_analysis_choice(self, event: Any) -> None:
        self.analysis_editor.rebuild(
            ANALYSIS_REGISTRY.get(event.new),
            channel_names=self._channel_names(),
        )

    def _on_run_analysis(self, *_: Any) -> None:
        try:
            key = str(self.analysis_select.value)
            cfg = AnalysisConfig(key, self.analysis_editor.values(), True)
            res = self.session.add_analysis(cfg)
            resolved_channel = self._resolve_channel_name(res.channel)
            if resolved_channel is not None:
                selected = self._selected_plot_channels()
                if resolved_channel not in selected:
                    selected = [resolved_channel]
                self.channel_select.value = selected
            self.status.object = (
                f"Ran `{ANALYSIS_REGISTRY.get(key).display_name}`; result "
                f"`{res.name}` has {len(res.arrays)} array field(s)."
            )
            self._refresh_results()
            options = list(self.result_select.options)
            if options:
                self.result_select.value = options[-1]
            self._update_plot()
        except Exception as exc:
            self._set_error(exc)

    def _on_rerun_analyses(self, *_: Any) -> None:
        try:
            self.session.rerun_analyses()
            self.status.object = "Reran saved analyses."
            self._refresh_results()
            self._update_plot()
        except Exception as exc:
            self._set_error(exc)

    def _on_clear_results(self, *_: Any) -> None:
        self.session.results = []
        self.session.analyses = []
        self._refresh_results()
        self._refresh_batch()
        self._update_plot()

    # ------------------------------------------------------------------
    # Export callbacks
    # ------------------------------------------------------------------
    def _on_export(self, *_: Any) -> None:
        try:
            written = self.session.export(self.export_dir.value)
            lines = [f"- `{key}`: `{path}`" for key, path in written.items()]
            self.export_status.object = "Exported:\n" + "\n".join(lines)
        except Exception as exc:
            self.export_status.object = f"**Export failed:** `{type(exc).__name__}: {exc}`"

    def _selected_result_index(self) -> int | None:
        value = self.result_select.value
        if value in (None, ""):
            return None
        return int(str(value).split(":", 1)[0])

    def _on_export_selected(self, *_: Any) -> None:
        try:
            import pandas as pd
            from ..analysis import peak_result_to_dataframe

            idx = self._selected_result_index()
            if idx is None or idx >= len(self.session.results):
                raise ValueError("Select an analysis result to export.")
            directory = self.session.output_path(self.export_dir.value)
            directory.mkdir(parents=True, exist_ok=True)
            result = self.session.results[idx]
            stem = f"result_{idx:03d}_{result.name}_{result.channel}".replace("/", "_")
            metrics = pd.DataFrame([result.metrics_row(self.session.current_state)])
            metrics_path = directory / f"{stem}_metrics.csv"
            metrics.to_csv(metrics_path, index=False)
            try:
                event_df = self._peak_dataframe_for_result(result)
                arrays = event_df if event_df is not None else result.arrays_frame(self.session.current_state)
            except Exception:
                arrays = result.arrays_frame(self.session.current_state)
            lines = [f"- `metrics`: `{metrics_path}`"]
            if not arrays.empty:
                arrays_path = directory / f"{stem}_arrays.csv"
                arrays.to_csv(arrays_path, index=False)
                lines.append(f"- `arrays`: `{arrays_path}`")
            self.export_status.object = "Exported selected result:\n" + "\n".join(lines)
        except Exception as exc:
            self.export_status.object = f"**Selected export failed:** `{type(exc).__name__}: {exc}`"

    # ------------------------------------------------------------------
    # Batch/multi-session callbacks
    # ------------------------------------------------------------------
    def _selected_batch_index(self) -> int | None:
        return _parse_option_index(self.batch_session_select.value)

    def _selected_batch_trace_indices(self) -> list[int]:
        collection = self.session.current_collection
        if collection is None:
            return []
        n = len(collection)
        selected = [
            idx for idx in (_parse_option_index(v) for v in (self.batch_trace_sessions.value or []))
            if idx is not None and 0 <= idx < n
        ]
        if selected:
            return selected

        group_cols = _split_groupby(self.batch_trace_groupby.value)
        group_value = str(self.batch_trace_group.value or "")
        if group_cols and group_value:
            out: list[int] = []
            from ..collection import metadata_value

            for i, state in enumerate(collection.states):
                key = " | ".join(str(metadata_value(state, col, "")) for col in group_cols)
                if key == group_value:
                    out.append(i)
            return out
        return list(range(n))

    def _selected_batch_collection_for_plot(self):
        from ..collection import PhotometryCollection

        collection = self.session.current_collection
        if collection is None:
            return None
        indices = self._selected_batch_trace_indices()
        if not indices:
            return None
        return PhotometryCollection([collection.states[i] for i in indices])

    def _refresh_batch_trace_groups(self) -> None:
        collection = self.session.current_collection
        group_cols = _split_groupby(self.batch_trace_groupby.value)
        options = {"All groups": ""}
        if collection is not None and group_cols:
            from ..collection import metadata_value

            values: list[str] = []
            for state in collection.states:
                key = " | ".join(str(metadata_value(state, col, "")) for col in group_cols)
                if key not in values:
                    values.append(key)
            options.update({v: v for v in sorted(values)})
        current = self.batch_trace_group.value
        self.batch_trace_group.options = options
        if current not in options.values():
            self.batch_trace_group.value = ""

    def _on_load_batch(self, *_: Any) -> None:
        try:
            directory = self.batch_dir_input.value.strip() or "."
            metadata = self.batch_metadata_path.value.strip() or None
            coll = self.session.load_batch(
                directory,
                pattern=self.batch_pattern.value.strip() or "**/*.doric",
                metadata_table=metadata,
                metadata_on=self.batch_metadata_on.value.strip() or "subject",
            )
            self.batch_status.object = f"Loaded {len(coll)} session(s) from `{directory}`."
            self._refresh_all()
        except Exception as exc:
            self.batch_status.object = f"**Batch load failed:** `{type(exc).__name__}: {exc}`"

    def _on_process_batch(self, *_: Any) -> None:
        try:
            coll = self.session.recompute_batch(continue_on_error=True)
            extra = f" Failed: {len(self.session.batch_errors)}." if self.session.batch_errors else ""
            self.batch_status.object = f"Processed {len(coll)} session(s).{extra}"
            self._refresh_all()
        except Exception as exc:
            self.batch_status.object = f"**Batch processing failed:** `{type(exc).__name__}: {exc}`"

    def _on_run_batch_selected_analysis(self, *_: Any) -> None:
        try:
            key = str(self.analysis_select.value)
            cfg = AnalysisConfig(key, self.analysis_editor.values(), True)
            report = self.session.add_batch_analysis(cfg, continue_on_error=True)
            self.batch_status.object = f"Ran `{ANALYSIS_REGISTRY.get(key).display_name}` on {len(report.reports)} session(s)."
            self._refresh_batch()
            self._update_plot()
        except Exception as exc:
            self.batch_status.object = f"**Batch analysis failed:** `{type(exc).__name__}: {exc}`"

    def _on_rerun_batch_analyses(self, *_: Any) -> None:
        try:
            report = self.session.run_batch_analyses(continue_on_error=True)
            self.batch_status.object = f"Reran saved analyses on {len(report.reports)} session(s)."
            self._refresh_batch()
            self._update_plot()
        except Exception as exc:
            self.batch_status.object = f"**Batch analysis failed:** `{type(exc).__name__}: {exc}`"

    def _on_open_batch_session(self, *_: Any) -> None:
        try:
            idx = self._selected_batch_index()
            if idx is None:
                raise ValueError("Select a session to inspect.")
            state = self.session.select_batch_session(idx, processed=True)
            self.status.object = f"Opened batch session {idx}: `{state.subject}`."
            self._refresh_all()
        except Exception as exc:
            self.batch_status.object = f"**Open session failed:** `{type(exc).__name__}: {exc}`"

    def _on_summarise_batch(self, *_: Any) -> None:
        try:
            vals = [v.strip() for v in self.batch_values.value.split(",") if v.strip()] or None
            grouped = self.session.batch_grouped_metrics(self.batch_groupby.value.strip() or None, vals)
            self.batch_grouped_table.value = grouped
            self._update_batch_summary_plot(grouped)
            self.batch_status.object = f"Computed grouped summary with {len(grouped)} row(s)."
        except Exception as exc:
            self.batch_status.object = f"**Batch summary failed:** `{type(exc).__name__}: {exc}`"

    def _on_export_batch(self, *_: Any) -> None:
        try:
            written = self.session.export_batch(self.export_dir.value)
            lines = [f"- `{key}`: `{path}`" for key, path in written.items()]
            self.batch_status.object = "Exported batch files:\n" + "\n".join(lines)
        except Exception as exc:
            self.batch_status.object = f"**Batch export failed:** `{type(exc).__name__}: {exc}`"

    def _update_batch_summary_plot(self, grouped: Any) -> None:
        from bokeh.models import ColumnDataSource, Whisker
        from bokeh.plotting import figure
        import pandas as pd

        df = pd.DataFrame(grouped)
        if df.empty or "mean" not in df.columns or "metric" not in df.columns:
            self.batch_plot_pane.object = self._empty_trace_figure("Run a grouped batch summary to plot metric means ± SEM.")
            return
        metric = str(df["metric"].iloc[0])
        df = df[df["metric"] == metric].copy()
        group_cols = [c for c in df.columns if c not in {"metric", "count", "mean", "std", "sem", "median", "min", "max", "q25", "q75"}]
        if group_cols:
            df["group"] = df[group_cols].astype(str).agg(" | ".join, axis=1)
        else:
            df["group"] = ["all"] * len(df)
        df["upper"] = df["mean"] + df.get("sem", 0)
        df["lower"] = df["mean"] - df.get("sem", 0)
        factors = df["group"].tolist()
        p = figure(
            x_range=factors,
            height=320,
            sizing_mode="stretch_width",
            title=f"Batch summary: {metric}",
            tools="pan,wheel_zoom,box_zoom,reset,save,crosshair",
            toolbar_location="above",
            y_axis_label=metric,
            x_axis_label=", ".join(group_cols) if group_cols else "group",
            output_backend=str(self.plot_backend.value or "svg"),
        )
        p.toolbar.logo = None
        p.grid.grid_line_color = "#e6edf5"
        source = ColumnDataSource(df)
        p.circle(x="group", y="mean", size=9, source=source, color=PLOT_PRIMARY, alpha=0.95)
        if "sem" in df.columns:
            whisker = Whisker(base="group", upper="upper", lower="lower", source=source)
            whisker.upper_head.size = 8
            whisker.lower_head.size = 8
            p.add_layout(whisker)
        p.xaxis.major_label_orientation = 0.65
        self.batch_plot_pane.object = p

    def _refresh_batch(self) -> None:
        summary = self.session.batch_summary_dataframe()
        self.batch_sessions_table.value = summary
        options = []
        if not summary.empty:
            for i, row in summary.iterrows():
                subject = row.get("subject", "")
                parent = row.get("source_parent", "")
                options.append(f"{i}: {subject} / {parent}")
        self.batch_session_select.options = options
        self.batch_trace_sessions.options = options
        if options and self.batch_session_select.value not in options:
            self.batch_session_select.value = options[0]
        elif not options:
            self.batch_session_select.value = None
        # Preserve valid trace multi-selections; blank means "all sessions".
        selected_trace = [v for v in (self.batch_trace_sessions.value or []) if v in options]
        if len(selected_trace) != len(self.batch_trace_sessions.value or []):
            self.batch_trace_sessions.value = selected_trace
        self._refresh_batch_trace_groups()
        metrics = self.session.batch_metrics_dataframe()
        self.batch_metrics_table.value = metrics.head(5000) if not metrics.empty else metrics
        if self.batch_groupby.value.strip() and not metrics.empty:
            try:
                grouped = self.session.batch_grouped_metrics(self.batch_groupby.value.strip())
                self.batch_grouped_table.value = grouped
                self._update_batch_summary_plot(grouped)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Refresh and plotting
    # ------------------------------------------------------------------
    def _refresh_all(self) -> None:
        self.history_policy.value = self.session.history_policy
        if self.output_dir_input.value != self.session.output_dir:
            self.output_dir_input.value = self.session.output_dir
        names = self._channel_names()
        self.channel_select.options = names
        if names:
            current_value = self.channel_select.value
            if isinstance(current_value, (list, tuple, set)):
                selected = [str(v) for v in current_value if str(v) in names]
            elif current_value in (None, ""):
                selected = []
            else:
                selected = [str(current_value)] if str(current_value) in names else []
            if not selected:
                selected = self._default_plot_channels(names)
            self.channel_select.value = selected
        else:
            self.channel_select.value = []
        self.stage_add_editor.update_channels(names)
        if str(self.stage_select.value) in {"IsosbesticRegression", "IsosbesticDff"}:
            self._on_stage_choice(type("Event", (), {"new": self.stage_select.value})())
        self.stage_edit_editor.update_channels(names)
        self.analysis_editor.update_channels(names)
        self._refresh_pipeline()
        self._refresh_results()
        self._refresh_batch()
        self._update_plot()

    def _refresh_pipeline(self) -> None:
        self.pipeline_table.value = self._pipeline_frame()
        options = [f"{i}: {cfg.name}" for i, cfg in enumerate(self.session.pipeline)]
        self.edit_stage_select.options = options
        if options and self.edit_stage_select.value not in options:
            self.edit_stage_select.value = options[-1]
        elif not options:
            self.edit_stage_select.value = None
        self._on_edit_stage_choice()

    def _refresh_results(self) -> None:
        n = len(self.session.results)
        self.results_summary.object = f"{n} analysis result(s)." if n else "No analyses run."
        options = [f"{i}: {r.name} / {r.channel}" for i, r in enumerate(self.session.results)]
        self.result_select.options = options
        if options and self.result_select.value not in options:
            self.result_select.value = options[-1]
        elif not options:
            self.result_select.value = None
        self.metrics_table.value = self.session.metrics_dataframe()
        arr = self.session.arrays_dataframe()
        self.arrays_table.value = arr.head(5000) if not arr.empty else arr

    def _resolve_channel_name(self, channel: str | None) -> str | None:
        if channel is None or self.session.current_state is None:
            return None
        return self._resolve_channel_from_names(
            channel,
            list(self.session.current_state.channel_names),
        )

    def _selected_result(self):
        idx = self._selected_result_index()
        if idx is not None and 0 <= idx < len(self.session.results):
            return self.session.results[idx]
        return self.session.results[-1] if self.session.results else None

    def _peak_dataframe_for_result(self, result: Any):
        """Return event coordinates for any peak-like result.

        Peak result names have changed over the package lifetime (`peaks`,
        `peak_analysis`, `peaks_by_template`).  Do not rely on the name here;
        instead, treat any analysis result with event arrays containing `x` and
        `y` as overlayable peak/event data.
        """
        if result is None:
            return None
        arrays = getattr(result, "arrays", {}) or {}
        if not {"x", "y"}.issubset(arrays):
            return None
        from ..analysis import peak_result_to_dataframe

        try:
            df = peak_result_to_dataframe(result)
        except Exception:
            return None
        if df.empty or not {"x", "y"}.issubset(df.columns):
            return None
        return df

    def _latest_peak_result(self):
        for res in reversed(self.session.results):
            if self._peak_dataframe_for_result(res) is not None:
                return res
        return None

    def _result_for_overlay(self):
        selected = self._selected_result()
        if selected is not None:
            if self._peak_dataframe_for_result(selected) is not None:
                return selected
            # Non-event selected results (e.g. QC/AUC) can still have window
            # overlays. Keep them selected rather than silently switching.
            arrays = getattr(selected, "arrays", {}) or {}
            if {"t_window", "y_window"}.issubset(arrays) or getattr(selected, "window", None) is not None:
                return selected
        return self._latest_peak_result()

    def _style_bokeh_trace(self, p: Any) -> Any:
        p.title.text_color = "#182433"
        p.title.text_font_size = "13pt"
        p.title.text_font_style = "normal"
        p.background_fill_color = "#ffffff"
        p.border_fill_color = "#ffffff"
        p.outline_line_color = "#dbe3ef"
        p.grid.grid_line_color = "#e6edf5"
        p.grid.grid_line_alpha = 0.85
        p.axis.axis_line_color = "#9aa8b8"
        p.axis.major_tick_line_color = "#9aa8b8"
        p.axis.minor_tick_line_color = None
        p.axis.major_label_text_color = "#344256"
        p.axis.axis_label_text_color = "#182433"
        p.toolbar.logo = None
        return p

    def _new_trace_figure(self, *, title: str, x_axis_label: str, y_axis_label: str):
        from bokeh.plotting import figure

        p = figure(
            min_height=470,
            sizing_mode="stretch_both",
            tools="pan,wheel_zoom,box_zoom,reset,save,crosshair",
            active_scroll="wheel_zoom",
            output_backend=str(self.plot_backend.value or "svg"),
            toolbar_location="above",
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            title=title,
        )
        return self._style_bokeh_trace(p)

    def _empty_trace_figure(self, message: str | None = None):
        from bokeh.models import Label

        p = self._new_trace_figure(
            title="fibphot trace viewer",
            x_axis_label="Time (s)",
            y_axis_label="Signal",
        )
        if message:
            p.add_layout(
                Label(
                    x=0,
                    y=0,
                    x_units="screen",
                    y_units="screen",
                    text=message,
                    text_color="#5e6b7a",
                    text_font_size="12pt",
                    x_offset=24,
                    y_offset=32,
                )
            )
        return p

    def _finish_plot(self, p: Any, hover_renderers: list[Any] | None = None) -> None:
        from bokeh.models import HoverTool

        hover_renderers = hover_renderers or []
        if hover_renderers:
            p.add_tools(
                HoverTool(
                    renderers=hover_renderers,
                    tooltips=[("time", "@time_s{0.000}"), ("signal", "@signal{0.0000}")],
                    mode="mouse",
                )
            )
        if p.legend:
            p.legend.location = "top_left"
            p.legend.click_policy = "mute"
            p.legend.background_fill_alpha = 0.78
            p.legend.border_line_alpha = 0.0
            p.legend.label_text_color = "#344256"
            p.legend.label_text_font_size = "8pt"
        self.plot_pane.object = p

    def _analysis_result_prefers_own_plot(self, result: Any) -> bool:
        arrays = getattr(result, "arrays", {}) or {}
        return bool(
            "aligned_traces" in arrays
            or "lag_s" in arrays
            or "granger_lag_s" in arrays
        )

    def _update_plot(self) -> None:
        """Refresh the main trace pane.

        The GUI has three plotting scopes:
        - the currently inspected session;
        - batch overlays/averages;
        - the selected analysis result, e.g. aligned epochs or connectivity.

        This method was accidentally dropped in the previous patch while the
        specialised plotting methods were retained, which caused the GUI to
        fail at start-up.  Keep this as the only callback target for widgets so
        future plot modes can be added without changing callback wiring.
        """
        try:
            scope = str(self.trace_scope.value or "single")

            if scope == "batch":
                self._update_batch_trace_plot()
                return

            if scope == "analysis":
                result = self._selected_result()
                if result is None:
                    self.plot_pane.object = self._empty_trace_figure(
                        "Run or select an analysis result to plot."
                    )
                    return
                if self._update_analysis_result_plot(result):
                    return
                # Some analyses are naturally overlaid on the session trace
                # rather than drawn as their own curves.
                self._update_single_trace_plot(result_override=result)
                return

            self._update_single_trace_plot()
        except Exception as exc:
            # Plotting errors should not kill the Panel session.  Show the
            # issue in-place so the user can adjust the offending option.
            self.plot_pane.object = self._empty_trace_figure(
                f"Plot update failed: {type(exc).__name__}: {exc}"
            )

    def _update_single_trace_plot(self, result_override: Any | None = None) -> None:
        """Plot one or more channels from the currently inspected session.

        The normal trace viewer now accepts a multi-selection of channels.  It
        overlays the selected processed traces, optionally overlays matching raw
        traces, and draws analysis/event markers on the relevant result channel
        rather than repeating the same markers on every displayed channel.
        """
        from bokeh.models import ColumnDataSource

        if self.session.current_state is None:
            self.plot_pane.object = self._empty_trace_figure(
                "Load a recording, or open a processed batch session, to view traces."
            )
            return

        channels = self._selected_plot_channels()
        if not channels:
            self.plot_pane.object = self._empty_trace_figure("Select one or more channels to plot.")
            return

        current = self.session.current_state
        resolved_channels: list[str] = []
        for channel in channels:
            if channel in current.channel_names:
                resolved_channels.append(channel)
                continue
            resolved = self._resolve_channel_name(channel)
            if resolved is not None:
                resolved_channels.append(resolved)
        if not resolved_channels:
            self.plot_pane.object = self._empty_trace_figure(
                "None of the selected channels are present in the current session."
            )
            return
        channels = resolved_channels

        result = result_override if result_override is not None else self._result_for_overlay()
        peak_df = self._peak_dataframe_for_result(result) if result is not None else None
        preserve_times = None
        if peak_df is not None and "x" in peak_df.columns:
            preserve_times = np.asarray(peak_df["x"], dtype=float)

        t = np.asarray(current.time_seconds, dtype=float)
        max_points = int(self.max_points.value or 20000)
        title_channels = ", ".join(channels[:4]) + (" …" if len(channels) > 4 else "")
        p = self._new_trace_figure(
            title=f"fibphot trace viewer — {title_channels}",
            x_axis_label="Time (s)",
            y_axis_label="Signal",
        )
        hover_renderers: list[Any] = []

        for ci, channel in enumerate(channels):
            if self.show_processed.value:
                y = np.asarray(current.channel(channel), dtype=float)
                tt, yy = _downsample(t, y, max_points, preserve_times=preserve_times)
                source = ColumnDataSource({"time_s": tt, "signal": yy, "channel": [channel] * len(tt)})
                r = p.line(
                    "time_s",
                    "signal",
                    source=source,
                    line_width=1.5 if len(channels) <= 3 else 1.25,
                    color=_channel_colour(channel, fallback_index=ci),
                    alpha=0.95,
                    legend_label=f"processed {channel}",
                    muted_alpha=0.12,
                )
                hover_renderers.append(r)

            if (
                self.show_raw.value
                and self.session.raw_state is not None
                and channel in self.session.raw_state.channel_names
            ):
                raw = self.session.raw_state
                tr = np.asarray(raw.time_seconds, dtype=float)
                yr = np.asarray(raw.channel(channel), dtype=float)
                tr, yr = _downsample(tr, yr, max_points, preserve_times=preserve_times)
                source = ColumnDataSource({"time_s": tr, "signal": yr, "channel": [channel] * len(tr)})
                r = p.line(
                    "time_s",
                    "signal",
                    source=source,
                    line_width=0.9,
                    color=_channel_raw_colour(channel, fallback_index=ci),
                    alpha=0.28,
                    legend_label=f"raw {channel}",
                    muted_alpha=0.06,
                )
                hover_renderers.append(r)

        if result is not None:
            result_channel = getattr(result, "channel", None)
            display_channel = None
            if result_channel is not None:
                display_channel = self._resolve_channel_from_names(str(result_channel), channels)
            if display_channel is None and len(channels) == 1:
                display_channel = channels[0]
            if display_channel is not None:
                self._add_result_overlay(
                    p,
                    result,
                    hover_renderers,
                    display_channel=display_channel,
                    peak_dataframe=peak_df,
                )

        self._finish_plot(p, hover_renderers)

    def _update_batch_trace_plot(self) -> None:
        from bokeh.models import ColumnDataSource

        collection = self._selected_batch_collection_for_plot()
        if collection is None or not len(collection):
            self.plot_pane.object = self._empty_trace_figure(
                "Load/process a batch, then choose batch sessions or a metadata group."
            )
            return
        channel = self._primary_plot_channel()
        if not channel:
            self.plot_pane.object = self._empty_trace_figure("Select a channel to plot.")
            return
        try:
            stats = collection.trace_statistics(channels=[channel], align="intersection", time_ref="start")
        except Exception as exc:
            self.plot_pane.object = self._empty_trace_figure(f"Could not align batch traces: {exc}")
            return

        x = np.asarray(stats.time_seconds, dtype=float)
        yi = np.asarray(stats.aligned[:, 0, :], dtype=float)
        mean = np.asarray(stats.mean[0], dtype=float)
        std = np.asarray(stats.std[0], dtype=float)
        sem = np.asarray(stats.sem[0], dtype=float)
        mode = str(self.batch_trace_mode.value or "both")
        err = str(self.batch_trace_error.value or "sem")
        max_points = int(self.max_points.value or 20000)

        p = self._new_trace_figure(
            title=f"Batch trace viewer — {len(collection)} session(s)",
            x_axis_label="Time from session start (s)",
            y_axis_label=channel,
        )
        hover_renderers: list[Any] = []

        if mode in {"overlay", "both"}:
            max_lines = min(yi.shape[0], 80)
            for si in range(max_lines):
                xx, yy = _downsample(x, yi[si], max_points)
                label = stats.subjects[si] or f"session {si}"
                source = ColumnDataSource({"time_s": xx, "signal": yy})
                r = p.line(
                    "time_s",
                    "signal",
                    source=source,
                    line_width=0.9,
                    color=_channel_raw_colour(channel),
                    alpha=0.24,
                    muted_alpha=0.05,
                    legend_label="individual sessions",
                )
                hover_renderers.append(r)
            if yi.shape[0] > max_lines:
                self.batch_status.object = (
                    f"Displayed {max_lines} of {yi.shape[0]} aligned sessions to keep the plot responsive."
                )

        if mode in {"mean", "both"}:
            xx, mm = _downsample(x, mean, max_points)
            if err != "none":
                ee_full = sem if err == "sem" else std
                _, ee = _downsample(x, ee_full, max_points)
                source_band = ColumnDataSource(
                    {"time_s": xx, "lower": mm - ee, "upper": mm + ee}
                )
                p.varea(
                    x="time_s",
                    y1="lower",
                    y2="upper",
                    source=source_band,
                    fill_color=_channel_colour(channel),
                    fill_alpha=0.16,
                    legend_label=f"mean ± {err.upper()}",
                )
            source = ColumnDataSource({"time_s": xx, "signal": mm})
            r = p.line(
                "time_s",
                "signal",
                source=source,
                line_width=2.6,
                color=_channel_colour(channel),
                alpha=0.98,
                legend_label="mean trace",
                muted_alpha=0.2,
            )
            hover_renderers.append(r)

        self._finish_plot(p, hover_renderers)

    def _update_analysis_result_plot(self, result: Any) -> bool:
        arrays = getattr(result, "arrays", {}) or {}
        mode = str(self.result_plot_mode.value or "auto")
        if mode in {"auto", "epochs"} and "aligned_traces" in arrays:
            self._plot_aligned_epochs_result(result)
            return True
        if mode in {"auto", "curve"} and (
            "lag_s" in arrays or "granger_lag_s" in arrays
        ):
            self._plot_curve_result(result)
            return True
        if mode == "trace":
            # fall through to the normal session trace overlay
            return False
        return False

    def _aligned_channel_indices(self, result: Any) -> list[tuple[int, str]]:
        arrays = getattr(result, "arrays", {}) or {}
        names = tuple(
            str(x) for x in np.asarray(arrays.get("aligned_channel_names", [])).tolist()
        )
        if not names:
            return []
        mode = str(self.aligned_channel_mode.value or "all")
        if mode == "all":
            return list(enumerate(names))
        requested_channels = self._selected_plot_channels()
        selected: list[tuple[int, str]] = []
        for requested in requested_channels:
            for i, name in enumerate(names):
                if name == requested or name.lower() == requested.lower():
                    selected.append((i, name))
                    break
        return selected or [(0, names[0])]

    def _plot_aligned_epochs_result(self, result: Any) -> None:
        from bokeh.models import ColumnDataSource

        arrays = getattr(result, "arrays", {}) or {}
        channel_info = self._aligned_channel_indices(result)
        if not channel_info:
            self.plot_pane.object = self._empty_trace_figure(
                "Selected result has no aligned channel names."
            )
            return
        t = np.asarray(arrays.get("time_relative_s"), dtype=float)
        data = np.asarray(arrays.get("aligned_traces"), dtype=float)
        if data.ndim != 3 or t.size != data.shape[2]:
            self.plot_pane.object = self._empty_trace_figure(
                "Selected result does not contain a valid aligned epoch stack."
            )
            return

        # Drop channel indices that are not valid for this result.
        channel_info = [(ci, ch) for ci, ch in channel_info if ci < data.shape[1]]
        if not channel_info:
            self.plot_pane.object = self._empty_trace_figure(
                "No selected aligned channels are present in this result."
            )
            return

        display = str(self.epoch_display.value or "both")
        err = str(self.epoch_error.value or "sem")
        max_points = int(self.max_points.value or 20000)
        all_channels = len(channel_info) > 1
        names = ", ".join(ch for _, ch in channel_info)

        p = self._new_trace_figure(
            title=f"{result.name}: peak-aligned epochs ({names}; n={data.shape[0]})",
            x_axis_label="Time relative to event peak (s)",
            y_axis_label="Signal",
        )
        hover_renderers: list[Any] = []

        finite = data[:, [ci for ci, _ in channel_info], :]
        y_min = float(np.nanmin(finite)) if np.isfinite(finite).any() else -1.0
        y_max = float(np.nanmax(finite)) if np.isfinite(finite).any() else 1.0
        p.line([0, 0], [y_min, y_max], color="#64748b", line_dash="dashed", alpha=0.55)

        for k, (ci, channel) in enumerate(channel_info):
            colour = _channel_colour(channel, fallback_index=k)
            y = data[:, ci, :]
            mean = np.nanmean(y, axis=0)
            std = np.nanstd(y, axis=0, ddof=1) if y.shape[0] > 1 else np.zeros_like(mean)
            sem = std / np.sqrt(max(y.shape[0], 1))
            q25 = np.nanpercentile(y, 25, axis=0)
            q75 = np.nanpercentile(y, 75, axis=0)

            if display in {"individual", "both"}:
                # Showing all epochs for all channels can become very dense.
                # Keep the plot responsive while still making raw event-level
                # variability visible.
                max_epochs = min(y.shape[0], 80 if all_channels else 120)
                for ei in range(max_epochs):
                    xx, yy = _downsample(t, y[ei], max_points)
                    source = ColumnDataSource({"time_s": xx, "signal": yy})
                    r = p.line(
                        "time_s",
                        "signal",
                        source=source,
                        line_width=0.7,
                        color=colour,
                        alpha=0.12 if all_channels else 0.22,
                        muted_alpha=0.025,
                        legend_label=f"{channel} epochs",
                    )
                    hover_renderers.append(r)

            if display in {"mean", "both"}:
                xx, mm = _downsample(t, mean, max_points)
                if err != "none":
                    if err == "sem":
                        _, ee = _downsample(t, sem, max_points)
                        lower, upper = mm - ee, mm + ee
                        label = f"{channel} mean ± SEM"
                    elif err == "std":
                        _, ee = _downsample(t, std, max_points)
                        lower, upper = mm - ee, mm + ee
                        label = f"{channel} mean ± SD"
                    else:
                        _, lo = _downsample(t, q25, max_points)
                        _, hi = _downsample(t, q75, max_points)
                        lower, upper = lo, hi
                        label = f"{channel} IQR"
                    band = ColumnDataSource(
                        {"time_s": xx, "lower": lower, "upper": upper}
                    )
                    p.varea(
                        x="time_s",
                        y1="lower",
                        y2="upper",
                        source=band,
                        fill_color=colour,
                        fill_alpha=0.12 if all_channels else 0.18,
                        legend_label=label,
                    )
                source = ColumnDataSource({"time_s": xx, "signal": mm})
                r = p.line(
                    "time_s",
                    "signal",
                    source=source,
                    line_width=2.8,
                    color=colour,
                    alpha=0.98,
                    legend_label=f"{channel} mean",
                    muted_alpha=0.2,
                )
                hover_renderers.append(r)

        self._finish_plot(p, hover_renderers)

    def _plot_curve_result(self, result: Any) -> None:
        from bokeh.models import ColumnDataSource

        arrays = getattr(result, "arrays", {}) or {}
        hover_renderers: list[Any] = []
        p = self._new_trace_figure(
            title=f"{result.name}: connectivity / lag curve",
            x_axis_label="Lag (s)",
            y_axis_label="Value",
        )

        if "lag_s" in arrays and "corr_mean_by_pair" in arrays:
            lag = np.asarray(arrays["lag_s"], dtype=float)
            means = np.asarray(arrays["corr_mean_by_pair"], dtype=float)
            sems = np.asarray(arrays.get("corr_sem_by_pair", []), dtype=float)
            stds = np.asarray(arrays.get("corr_std_by_pair", []), dtype=float)
            by_event = np.asarray(arrays.get("corr_by_pair_event", []), dtype=float)
            labels = [
                str(x)
                for x in np.asarray(
                    arrays.get("connectivity_pair_labels", [f"pair {i}" for i in range(means.shape[0])])
                ).tolist()
            ]
            display = str(self.epoch_display.value or "both")
            err = str(self.epoch_error.value or "sem")

            for pi in range(means.shape[0]):
                colour = PLOT_CHANNELS[pi % len(PLOT_CHANNELS)]
                label = labels[pi] if pi < len(labels) else f"pair {pi}"
                if display in {"individual", "both"} and by_event.ndim == 3 and pi < by_event.shape[0]:
                    for i in range(min(by_event.shape[1], 80)):
                        y = by_event[pi, i]
                        if not np.isfinite(y).any():
                            continue
                        source = ColumnDataSource({"time_s": lag, "signal": y})
                        r = p.line(
                            "time_s",
                            "signal",
                            source=source,
                            color=colour,
                            alpha=0.10,
                            line_width=0.75,
                            legend_label=f"{label} event correlations",
                        )
                        hover_renderers.append(r)
                if display in {"mean", "both"}:
                    mean = means[pi]
                    if err != "none":
                        if err == "std" and stds.ndim == 2 and pi < stds.shape[0]:
                            e = stds[pi]
                            label_err = f"{label} mean ± SD"
                        elif sems.ndim == 2 and pi < sems.shape[0]:
                            e = sems[pi]
                            label_err = f"{label} mean ± SEM"
                        else:
                            e = np.zeros_like(mean)
                            label_err = f"{label} mean"
                        p.varea(
                            x=lag,
                            y1=mean - e,
                            y2=mean + e,
                            fill_color=colour,
                            fill_alpha=0.11,
                            legend_label=label_err,
                        )
                    source = ColumnDataSource({"time_s": lag, "signal": mean})
                    r = p.line(
                        "time_s",
                        "signal",
                        source=source,
                        color=colour,
                        line_width=2.6,
                        legend_label=f"{label} mean correlation",
                    )
                    hover_renderers.append(r)
            self._finish_plot(p, hover_renderers)
            return

        if "lag_s" in arrays and "corr_mean" in arrays:
            lag = np.asarray(arrays["lag_s"], dtype=float)
            mean = np.asarray(arrays["corr_mean"], dtype=float)
            corr_by_event = np.asarray(arrays.get("corr_by_event", []), dtype=float)
            display = str(self.epoch_display.value or "both")
            err = str(self.epoch_error.value or "sem")
            if display in {"individual", "both"} and corr_by_event.ndim == 2:
                for i in range(min(corr_by_event.shape[0], 120)):
                    source = ColumnDataSource({"time_s": lag, "signal": corr_by_event[i]})
                    r = p.line("time_s", "signal", source=source, color=PLOT_RAW, alpha=0.20, line_width=0.8, legend_label="event correlations")
                    hover_renderers.append(r)
            if display in {"mean", "both"}:
                if err != "none":
                    if err == "std" and "corr_std" in arrays:
                        e = np.asarray(arrays["corr_std"], dtype=float)
                        label = "mean ± SD"
                    elif "corr_sem" in arrays:
                        e = np.asarray(arrays["corr_sem"], dtype=float)
                        label = "mean ± SEM"
                    else:
                        e = np.zeros_like(mean)
                        label = "mean"
                    p.varea(x=lag, y1=mean - e, y2=mean + e, fill_color=PLOT_SHADE, fill_alpha=0.18, legend_label=label)
                source = ColumnDataSource({"time_s": lag, "signal": mean})
                r = p.line("time_s", "signal", source=source, color=PLOT_MEAN, line_width=2.7, legend_label="mean correlation")
                hover_renderers.append(r)
            self._finish_plot(p, hover_renderers)
            return

        if "lag_s" in arrays and "correlation" in arrays:
            lag = np.asarray(arrays["lag_s"], dtype=float)
            y = np.asarray(arrays["correlation"], dtype=float)
            source = ColumnDataSource({"time_s": lag, "signal": y})
            r = p.line("time_s", "signal", source=source, color=PLOT_MEAN, line_width=2.4, legend_label="correlation")
            hover_renderers.append(r)
            self._finish_plot(p, hover_renderers)
            return

        if "granger_lag_s" in arrays:
            lag = np.asarray(arrays["granger_lag_s"], dtype=float)
            plotted = False
            for key, label, colour in (
                ("p_x_to_y_mean", "x → y mean p", PLOT_MEAN),
                ("p_y_to_x_mean", "y → x mean p", PLOT_PEAKS),
            ):
                if key not in arrays:
                    continue
                y = np.asarray(arrays[key], dtype=float)
                m = np.isfinite(lag) & np.isfinite(y)
                if not np.any(m):
                    continue
                source = ColumnDataSource({"time_s": lag[m], "signal": y[m]})
                r = p.line("time_s", "signal", source=source, color=colour, line_width=2.4, legend_label=label)
                p.circle("time_s", "signal", source=source, color=colour, size=5, alpha=0.75, legend_label=label)
                hover_renderers.append(r)
                plotted = True
            if not plotted:
                p.text(
                    x=[0],
                    y=[0.5],
                    text=["No finite Granger p-values were returned. Try a shorter window, larger target_dt, lower max_lag_steps, or disable differencing."],
                    text_align="center",
                    text_baseline="middle",
                    text_color="#5e6b7a",
                )
            self._finish_plot(p, hover_renderers)
            return

        if "lag_s" in arrays and "p_x_to_y" in arrays:
            lag = np.asarray(arrays["lag_s"], dtype=float)
            plotted = False
            for key, label, colour in (
                ("p_x_to_y", "x → y p", PLOT_MEAN),
                ("p_y_to_x", "y → x p", PLOT_PEAKS),
            ):
                if key not in arrays:
                    continue
                y = np.asarray(arrays[key], dtype=float)
                m = np.isfinite(lag) & np.isfinite(y)
                if not np.any(m):
                    continue
                source = ColumnDataSource({"time_s": lag[m], "signal": y[m]})
                r = p.line("time_s", "signal", source=source, color=colour, line_width=2.4, legend_label=label)
                p.circle("time_s", "signal", source=source, color=colour, size=5, alpha=0.75, legend_label=label)
                hover_renderers.append(r)
                plotted = True
            if not plotted:
                p.text(
                    x=[0],
                    y=[0.5],
                    text=["No finite Granger p-values were returned. Try a shorter window, larger target_dt, lower max_lag_steps, or disable differencing."],
                    text_align="center",
                    text_baseline="middle",
                    text_color="#5e6b7a",
                )
            self._finish_plot(p, hover_renderers)
            return

        self.plot_pane.object = self._empty_trace_figure("Selected result has no plottable lag/curve arrays.")

    def _add_result_overlay(
        self,
        p: Any,
        result: Any,
        hover_renderers: list[Any],
        *,
        display_channel: str | None = None,
        peak_dataframe: Any | None = None,
    ) -> None:
        """Draw the selected analysis result on the interactive trace plot."""
        from bokeh.models import BoxAnnotation, ColumnDataSource

        # Window-like analyses, e.g. AUC, should visibly mark the analysed range.
        t0 = result.metrics.get("t0") if hasattr(result, "metrics") else None
        t1 = result.metrics.get("t1") if hasattr(result, "metrics") else None
        if t0 is None and getattr(result, "window", None) is not None:
            window = result.window
            if getattr(window, "ref", None) == "seconds":
                t0 = window.start
                t1 = window.end
        if t0 is not None and t1 is not None:
            try:
                p.add_layout(
                    BoxAnnotation(
                        left=float(t0),
                        right=float(t1),
                        fill_color="#15A3A3",
                        fill_alpha=0.07,
                        line_alpha=0.0,
                    )
                )
            except Exception:
                pass

        arrays = getattr(result, "arrays", {}) or {}
        if "t_window" in arrays and "y_window" in arrays:
            try:
                source = ColumnDataSource(
                    {
                        "time_s": np.asarray(arrays["t_window"], dtype=float),
                        "signal": np.asarray(arrays["y_window"], dtype=float),
                    }
                )
                r = p.line(
                    "time_s",
                    "signal",
                    source=source,
                    line_width=2.6,
                    color=_channel_colour(getattr(result, "channel", None)),
                    alpha=0.95,
                    legend_label=f"selected {result.name}",
                    muted_alpha=0.2,
                )
                hover_renderers.append(r)
            except Exception:
                pass

        if self.show_peaks.value:
            df = peak_dataframe if peak_dataframe is not None else self._peak_dataframe_for_result(result)
            if df is not None:
                event_times = np.asarray(df["x"], dtype=float)
                event_y = np.asarray(df["y"], dtype=float)
                if display_channel is not None and self.session.current_state is not None:
                    try:
                        trace_t = np.asarray(self.session.current_state.time_seconds, dtype=float)
                        trace_y = np.asarray(
                            self.session.current_state.channel(display_channel),
                            dtype=float,
                        )
                        event_y = _values_at_times(trace_t, trace_y, event_times)
                    except Exception:
                        pass
                data = {
                    "time_s": event_times,
                    "signal": event_y,
                }
                if "height" in df.columns:
                    data["height"] = np.asarray(df["height"], dtype=float)
                if "prominence" in df.columns:
                    data["prominence"] = np.asarray(df["prominence"], dtype=float)
                if "match_score" in df.columns:
                    data["match_score"] = np.asarray(df["match_score"], dtype=float)
                source = ColumnDataSource(data)
                r = p.scatter(
                    "time_s",
                    "signal",
                    source=source,
                    size=10,
                    marker="circle",
                    fill_color=PLOT_PEAKS,
                    line_color="#7f1d2d",
                    line_width=0.8,
                    alpha=0.95,
                    legend_label=f"{result.name} events",
                )
                hover_renderers.append(r)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def panel(self):
        pn = self.pn
        self.plot_pane.sizing_mode = "stretch_both"
        self.metrics_table.sizing_mode = "stretch_width"
        self.arrays_table.sizing_mode = "stretch_width"
        button_row_style = {"flex-wrap": "wrap", "gap": "0.35rem"}

        file_panel = pn.Card(
            pn.Row(self.path_input, self.load_button, sizing_mode="stretch_width"),
            self.browse_input,
            self.input_file_selector,
            pn.Row(
                self.output_dir_input,
                self.output_dir_button,
                sizing_mode="stretch_width",
                styles=button_row_style,
            ),
            pn.pane.Markdown(
                "<small>Relative export, pipeline and GUI-session paths are written under this output directory. "
                "By default this is the directory where the GUI was launched.</small>",
                margin=(0, 0, 4, 0),
            ),
            pn.Row(
                self.session_file_path,
                self.save_session_button,
                self.load_session_button,
                sizing_mode="stretch_width",
                styles=button_row_style,
            ),
            pn.pane.Markdown(
                "<small>A saved fibphot GUI session stores paths, output directory, pipeline, analyses, "
                "current processed data and results so it can be reopened later.</small>",
                margin=(0, 0, 4, 0),
            ),
            self.status,
            title="Session",
            collapsed=False,
            sizing_mode="stretch_width",
            css_classes=["fibphot-card"],
        )
        settings_panel = pn.Card(
            self.sidebar_width,
            self.history_policy,
            self.plot_backend,
            pn.pane.Markdown(
                "<small>SVG keeps traces crisp when resized; Canvas is faster for very dense traces.</small>",
                margin=(0, 0, 4, 0),
            ),
            title="Settings",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["fibphot-card"],
        )
        stage_add = pn.Card(
            self.stage_select,
            self.stage_add_editor.container,
            self.add_stage_button,
            title="Add processing stage",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["fibphot-card"],
        )
        stage_edit = pn.Card(
            self.pipeline_table,
            self.edit_stage_select,
            self.edit_stage_enabled,
            self.stage_edit_editor.container,
            pn.Row(self.apply_stage_button, self.remove_stage_button, sizing_mode="stretch_width", styles=button_row_style),
            pn.Row(self.stage_up_button, self.stage_down_button, self.undo_button, self.redo_button, self.recompute_button, sizing_mode="stretch_width", styles=button_row_style),
            pn.Row(self.pipeline_json_path, self.save_pipeline_button, self.load_pipeline_button, sizing_mode="stretch_width", styles=button_row_style),
            title="Pipeline",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["fibphot-card"],
        )
        analysis_actions = pn.Row(
            self.run_analysis_button,
            self.rerun_analyses_button,
            self.clear_results_button,
            sizing_mode="stretch_width",
            styles={
                **button_row_style,
                "position": "sticky",
                "top": "0",
                "z-index": "2",
                "background": "#ffffff",
                "padding": "0.35rem 0",
                "border-bottom": "1px solid #dbe3ef",
            },
        )
        analysis_params_box = pn.Column(
            self.analysis_editor.container,
            sizing_mode="stretch_width",
            styles={
                "max-height": "48vh",
                "overflow-y": "auto",
                "padding-right": "0.15rem",
            },
        )
        analysis_panel = pn.Card(
            self.analysis_select,
            analysis_actions,
            analysis_params_box,
            self.results_summary,
            self.result_select,
            title="Analysis",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["fibphot-card"],
        )
        batch_panel = pn.Card(
            self.batch_dir_input,
            self.batch_pattern,
            self.batch_metadata_path,
            self.batch_metadata_on,
            pn.Row(self.load_batch_button, self.process_batch_button, sizing_mode="stretch_width", styles=button_row_style),
            pn.Row(self.run_batch_selected_button, self.rerun_batch_button, sizing_mode="stretch_width", styles=button_row_style),
            self.batch_session_select,
            pn.Row(self.open_batch_session_button, self.export_batch_button, sizing_mode="stretch_width", styles=button_row_style),
            pn.pane.Markdown("**Batch trace overlay / average**", margin=(8, 0, 0, 0)),
            self.batch_trace_sessions,
            self.batch_trace_groupby,
            self.batch_trace_group,
            self.batch_groupby,
            self.batch_values,
            self.summarise_batch_button,
            self.batch_status,
            title="Batch / multi-session",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["fibphot-card"],
        )

        export_panel = pn.Card(
            self.export_dir,
            pn.pane.Markdown(
                "<small>Use a relative folder name to export under the Session output directory, "
                "or enter an absolute path to override it.</small>",
                margin=(0, 0, 4, 0),
            ),
            pn.Row(self.export_button, self.export_selected_button, sizing_mode="stretch_width", styles=button_row_style),
            self.export_status,
            title="Export",
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["fibphot-card"],
        )

        trace_controls = pn.Column(
            pn.Row(
                self.trace_scope,
                self.channel_select,
                self.show_raw,
                self.show_processed,
                self.show_peaks,
                self.max_points,
                sizing_mode="stretch_width",
                styles={"flex-wrap": "wrap", "gap": "0.5rem"},
            ),
            pn.Row(
                self.batch_trace_mode,
                self.batch_trace_error,
                self.result_plot_mode,
                self.epoch_display,
                self.epoch_error,
                self.aligned_channel_mode,
                sizing_mode="stretch_width",
                styles={"flex-wrap": "wrap", "gap": "0.5rem"},
            ),
            sizing_mode="stretch_width",
        )
        plot_panel = pn.Card(
            trace_controls,
            self.plot_pane,
            title="Trace viewer",
            collapsed=False,
            sizing_mode="stretch_both",
            css_classes=["fibphot-card", "fibphot-resizable-y"],
            styles={
                "resize": "vertical",
                "overflow": "auto",
                "min-height": "340px",
                "height": "520px",
                "max-height": "85vh",
            },
        )

        results_panel = pn.Tabs(
            ("Metrics", self.metrics_table),
            ("Event/array table", self.arrays_table),
            ("Batch sessions", self.batch_sessions_table),
            ("Batch metrics", self.batch_metrics_table),
            ("Batch grouped summary", self.batch_grouped_table),
            ("Batch plot", self.batch_plot_pane),
            sizing_mode="stretch_width",
            css_classes=["fibphot-card", "fibphot-resizable-y"],
            styles={
                "resize": "vertical",
                "overflow": "auto",
                "min-height": "180px",
                "height": "310px",
                "max-height": "75vh",
            },
        )

        sidebar = pn.Column(
            file_panel,
            settings_panel,
            stage_add,
            stage_edit,
            analysis_panel,
            batch_panel,
            export_panel,
            sizing_mode="stretch_height",
            width=int(self.sidebar_width.value),
            min_width=320,
            css_classes=["fibphot-sidebar"],
            styles={
                "width": f"{int(self.sidebar_width.value)}px",
            },
        )
        self._sidebar_container = sidebar

        main = pn.Column(
            plot_panel,
            results_panel,
            sizing_mode="stretch_both",
            min_width=520,
            css_classes=["fibphot-main"],
        )

        header = pn.Row(
            pn.pane.Markdown(
                "## fibphot GUI\n<p>Interactive fibre-photometry processing, analysis and export</p>",
                margin=(0, 12, 0, 0),
                css_classes=["fibphot-title"],
            ),
            pn.Spacer(sizing_mode="stretch_width"),
            pn.pane.Markdown(
                "Pan, wheel-zoom, box-zoom and save are available in the trace toolbar.",
                margin=(9, 0, 0, 0),
                css_classes=["fibphot-muted"],
            ),
            sizing_mode="stretch_width",
            css_classes=["fibphot-header"],
        )

        return pn.Column(
            header,
            pn.Row(sidebar, main, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
            css_classes=["fibphot-app"],
            styles={
                "height": "100vh",
                "width": "100vw",
                "overflow": "hidden",
            },
        )


def make_app():
    return FibPhotGUI().panel()


def main() -> None:
    pn = _require_panel()
    pn.serve(make_app, title="fibphot GUI", show=True)


if __name__.startswith("bokeh"):
    make_app().servable()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pandas as pd

from ..analysis import (
    AnalysisResult,
    PhotometryReport,
    grouped_numeric_summary,
    peak_result_to_dataframe,
    peak_triggered_summary_dataframe,
    peak_triggered_traces_dataframe,
)
from ..io import read_doric
from ..io.h5 import (
    _decode_str,
    _json_dumps,
    _json_loads,
    _read_payload_group,
    _read_state_group,
    _require_h5py,
    _write_payload_group,
    _write_state_group,
    load_state_h5,
    save_state_h5,
)
from ..collection import BatchReport, PhotometryCollection
from ..state import HistoryPolicy, PhotometryState
from .registry import AnalysisConfig, StageConfig, create_analysis, create_stage


def _analysis_result_to_payload(result: AnalysisResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "channel": result.channel,
        "window": None if result.window is None else result.window.as_dict(),
        "params": result.params,
        "metrics": result.metrics,
        "arrays": result.arrays,
        "notes": result.notes,
    }


def _window_from_payload(value: Any):
    from ..analysis import AnalysisWindow

    if value in (None, ""):
        return None
    if isinstance(value, AnalysisWindow):
        return value
    if isinstance(value, dict):
        return AnalysisWindow(
            start=value.get("start", 0),
            end=value.get("end", 0),
            ref=value.get("ref", "seconds"),
            label=value.get("label"),
        )
    return None


def _analysis_result_from_payload(payload: Mapping[str, Any]) -> AnalysisResult:
    return AnalysisResult(
        name=str(payload.get("name", "analysis")),
        channel=str(payload.get("channel", "")),
        window=_window_from_payload(payload.get("window")),
        params=dict(payload.get("params") or {}),
        metrics=dict(payload.get("metrics") or {}),
        arrays=dict(payload.get("arrays") or {}),
        notes=payload.get("notes"),
    )


def _write_results_group(
    group: Any,
    results: Sequence[AnalysisResult],
    *,
    compression: str | None,
    compression_opts: int,
) -> None:
    for i, result in enumerate(results):
        _write_payload_group(
            group.create_group(f"result_{i:04d}"),
            _analysis_result_to_payload(result),
            compression=compression,
            compression_opts=compression_opts,
        )


def _read_results_group(group: Any) -> list[AnalysisResult]:
    results: list[AnalysisResult] = []
    for key in sorted(group.keys()):
        payload = _read_payload_group(group[key])
        if isinstance(payload, Mapping):
            results.append(_analysis_result_from_payload(payload))
    return results


def _read_state(path: str | Path) -> PhotometryState:
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_state_h5(path)
    if path.suffix.lower() == ".doric":
        return read_doric(path)
    raise ValueError(
        f"Unsupported input file type: {path.suffix!r}. Use .doric, .h5 or .hdf5."
    )


@dataclass(slots=True)
class GuiSession:
    raw_state: PhotometryState | None = None
    current_state: PhotometryState | None = None
    source_path: str | None = None
    pipeline: list[StageConfig] = field(default_factory=list)
    analyses: list[AnalysisConfig] = field(default_factory=list)
    results: list[AnalysisResult] = field(default_factory=list)
    history_policy: HistoryPolicy = "raw"
    output_dir: str = field(default_factory=lambda: str(Path.cwd()))
    errors: list[str] = field(default_factory=list)

    # Batch/multi-session state.  These reuse the same pipeline and analysis
    # configs as the single-session workflow.
    raw_collection: PhotometryCollection | None = None
    current_collection: PhotometryCollection | None = None
    batch_report: BatchReport | None = None
    batch_source_dir: str | None = None
    batch_errors: list[str] = field(default_factory=list)

    _undo_stack: list[list[StageConfig]] = field(
        default_factory=list, repr=False
    )
    _redo_stack: list[list[StageConfig]] = field(
        default_factory=list, repr=False
    )

    @property
    def loaded(self) -> bool:
        return self.raw_state is not None

    @property
    def state(self) -> PhotometryState:
        if self.current_state is None:
            raise RuntimeError("No state is loaded.")
        return self.current_state

    def set_output_dir(self, path: str | Path) -> Path:
        """Set the root directory used for GUI outputs.

        Relative export paths, pipeline JSON paths, and selected-result exports
        are resolved under this directory.  The default is the working directory
        from which the GUI process was launched.
        """
        out = Path(path).expanduser()
        if not out.is_absolute():
            out = Path.cwd() / out
        out.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(out)
        return out

    def output_path(self, path: str | Path) -> Path:
        """Resolve a user-supplied path relative to ``output_dir``.

        Absolute paths are preserved.  Relative paths are placed underneath the
        GUI output directory.
        """
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        return Path(self.output_dir).expanduser() / p

    def load(self, path: str | Path) -> PhotometryState:
        state = _read_state(path)
        self.raw_state = state
        self.current_state = state
        self.source_path = str(path)
        self.pipeline = []
        self.analyses = []
        self.results = []
        self.errors = []
        self._undo_stack = []
        self._redo_stack = []
        return state

    def _snapshot(self) -> None:
        self._undo_stack.append(
            [replace(c, params=dict(c.params)) for c in self.pipeline]
        )
        self._redo_stack.clear()

    def _compute_pipeline(self, pipeline: list[StageConfig]) -> PhotometryState:
        if self.raw_state is None:
            raise RuntimeError("No state is loaded.")
        state = self.raw_state
        errors: list[str] = []
        for i, cfg in enumerate(pipeline):
            if not cfg.enabled:
                continue
            try:
                stage = create_stage(cfg)
                state = state.pipe(stage, history=self.history_policy)
            except Exception as exc:
                msg = f"Stage {i + 1} ({cfg.name}) failed: {exc}"
                errors.append(msg)
                self.errors = errors
                raise RuntimeError(msg) from exc
        self.errors = []
        return state

    def set_pipeline(
        self, pipeline: list[StageConfig], *, snapshot: bool = True
    ) -> None:
        candidate = [replace(c, params=dict(c.params)) for c in pipeline]
        # Transactional update: validate/recompute the candidate pipeline first.
        # If it fails, keep the previous pipeline/current_state/results intact so
        # the GUI can display the error without forcing a session reload.
        new_state = self._compute_pipeline(candidate)
        if snapshot:
            self._snapshot()
        self.pipeline = candidate
        self.current_state = new_state
        self.results = []

    def add_stage(self, config: StageConfig) -> None:
        self.set_pipeline([*self.pipeline, config])

    def update_stage(self, index: int, config: StageConfig) -> None:
        p = list(self.pipeline)
        p[index] = config
        self.set_pipeline(p)

    def remove_stage(self, index: int) -> None:
        p = list(self.pipeline)
        del p[index]
        self.set_pipeline(p)

    def move_stage(self, index: int, delta: int) -> None:
        j = index + delta
        if j < 0 or j >= len(self.pipeline):
            return
        p = list(self.pipeline)
        p[index], p[j] = p[j], p[index]
        self.set_pipeline(p)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(
            [replace(c, params=dict(c.params)) for c in self.pipeline]
        )
        self.pipeline = self._undo_stack.pop()
        self.recompute()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(
            [replace(c, params=dict(c.params)) for c in self.pipeline]
        )
        self.pipeline = self._redo_stack.pop()
        self.recompute()
        return True

    def recompute(self) -> PhotometryState:
        state = self._compute_pipeline(self.pipeline)
        self.current_state = state
        self.results = []
        return state

    def add_analysis(self, config: AnalysisConfig) -> AnalysisResult:
        if self.current_state is None:
            raise RuntimeError("No state is loaded.")
        self.analyses.append(config)
        analysis = create_analysis(config)
        result = analysis(self.current_state)
        self.results.append(result)
        return result

    def rerun_analyses(self) -> list[AnalysisResult]:
        if self.current_state is None:
            raise RuntimeError("No state is loaded.")
        out: list[AnalysisResult] = []
        for cfg in self.analyses:
            if not cfg.enabled:
                continue
            out.append(create_analysis(cfg)(self.current_state))
        self.results = out
        return out

    def metrics_dataframe(self) -> pd.DataFrame:
        if self.current_state is None or not self.results:
            return pd.DataFrame()
        return PhotometryReport(
            self.current_state, tuple(self.results)
        ).metrics_dataframe()

    def arrays_dataframe(self) -> pd.DataFrame:
        if self.current_state is None or not self.results:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for result in self.results:
            if result.name in {"peak_analysis", "peaks_by_template", "peaks"}:
                try:
                    frames.append(peak_result_to_dataframe(result))
                    continue
                except Exception:
                    pass
            if result.name == "peak_triggered_average":
                try:
                    frames.append(peak_triggered_summary_dataframe(result))
                    continue
                except Exception:
                    pass
            frames.append(result.arrays_frame(self.current_state))
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    # ------------------------------------------------------------------
    # Batch/multi-session workflow
    # ------------------------------------------------------------------
    def load_batch(
        self,
        directory: str | Path,
        *,
        pattern: str = "**/*.doric",
        metadata_table: str | Path | None = None,
        metadata_on: str = "subject",
    ) -> PhotometryCollection:
        directory = Path(directory).expanduser()
        collection = PhotometryCollection.from_directory(
            directory, pattern=pattern, reader=read_doric
        )
        if metadata_table:
            collection = collection.with_metadata_table(
                metadata_table, on=metadata_on
            )
        self.raw_collection = collection
        self.current_collection = collection
        self.batch_report = None
        self.batch_source_dir = str(directory)
        self.batch_errors = []
        return collection

    def recompute_batch(
        self, *, continue_on_error: bool = True
    ) -> PhotometryCollection:
        if self.raw_collection is None:
            raise RuntimeError("No batch collection is loaded.")
        stages = [create_stage(cfg) for cfg in self.pipeline if cfg.enabled]
        result = self.raw_collection.batch_pipe(
            *stages,
            history=self.history_policy,
            continue_on_error=continue_on_error,
        )
        self.current_collection = result.successful
        self.batch_errors = [
            f"{f.subject or f.index}: {f.step}: {f.error_type}: {f.message}"
            for f in result.failed
        ]
        self.batch_report = None
        return result.successful

    def run_batch_analyses(
        self, *, continue_on_error: bool = True
    ) -> BatchReport:
        if self.current_collection is None:
            raise RuntimeError("No processed batch collection is available.")
        analyses = [
            create_analysis(cfg) for cfg in self.analyses if cfg.enabled
        ]
        if not analyses:
            raise RuntimeError(
                "No saved analyses are configured. Run or add an analysis first."
            )
        report = self.current_collection.analyse(
            *analyses, continue_on_error=continue_on_error
        )
        self.batch_report = report
        self.batch_errors = [
            f"{f.subject or f.index}: {f.step}: {f.error_type}: {f.message}"
            for f in report.failed
        ]
        return report

    def add_batch_analysis(
        self, config: AnalysisConfig, *, continue_on_error: bool = True
    ) -> BatchReport:
        if self.current_collection is None:
            raise RuntimeError("No processed batch collection is available.")
        self.analyses.append(config)
        report = self.current_collection.analyse(
            create_analysis(config), continue_on_error=continue_on_error
        )
        if self.batch_report is None:
            self.batch_report = report
        else:
            self.batch_report = BatchReport(
                reports=(*self.batch_report.reports, *report.reports),
                failed=(*self.batch_report.failed, *report.failed),
            )
        self.batch_errors = [
            f"{f.subject or f.index}: {f.step}: {f.error_type}: {f.message}"
            for f in report.failed
        ]
        return report

    def batch_summary_dataframe(self) -> pd.DataFrame:
        if self.current_collection is None:
            return pd.DataFrame()
        return self.current_collection.summary_table()

    def batch_metrics_dataframe(self) -> pd.DataFrame:
        if self.batch_report is None:
            return pd.DataFrame()
        return self.batch_report.metrics_dataframe()

    def batch_arrays_dataframe(self) -> pd.DataFrame:
        if self.batch_report is None:
            return pd.DataFrame()
        return self.batch_report.arrays_dataframe()

    def batch_grouped_metrics(
        self, groupby: str | None, values: list[str] | None = None
    ) -> pd.DataFrame:
        df = self.batch_metrics_dataframe()
        return (
            grouped_numeric_summary(df, groupby=groupby, values=values)
            if not df.empty
            else df
        )

    def select_batch_session(
        self, index: int, *, processed: bool = True
    ) -> PhotometryState:
        collection = (
            self.current_collection if processed else self.raw_collection
        )
        if collection is None:
            raise RuntimeError("No batch collection is loaded.")
        state = collection.states[int(index)]
        self.raw_state = (
            self.raw_collection.states[int(index)]
            if self.raw_collection is not None
            and int(index) < len(self.raw_collection.states)
            else state
        )
        self.current_state = state
        self.source_path = str(state.metadata.get("source_path", ""))
        self.results = []
        if self.batch_report is not None and int(index) < len(
            self.batch_report.reports
        ):
            self.results = list(self.batch_report.reports[int(index)].results)
        return state

    def export_batch(self, directory: str | Path) -> dict[str, Path]:
        directory = self.output_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        summary = self.batch_summary_dataframe()
        if not summary.empty:
            p = directory / "batch_sessions.csv"
            summary.to_csv(p, index=False)
            written["batch_sessions"] = p
        metrics = self.batch_metrics_dataframe()
        if not metrics.empty:
            p = directory / "batch_analysis_metrics.csv"
            metrics.to_csv(p, index=False)
            written["batch_analysis_metrics"] = p
        arrays = self.batch_arrays_dataframe()
        if not arrays.empty:
            p = directory / "batch_analysis_arrays.csv"
            arrays.to_csv(p, index=False)
            written["batch_analysis_arrays"] = p
        return written

    def save_session_h5(
        self,
        path: str | Path,
        *,
        include_raw: bool = True,
        include_batch: bool = True,
        compression: str | None = "gzip",
        compression_opts: int = 4,
    ) -> Path:
        """Save the full GUI working session to a compact HDF5 file.

        This stores the currently loaded paths/directories, output directory,
        processing pipeline, saved analysis configurations, current processed
        state, analysis results, and, optionally, loaded batch collections and
        batch results.  It is intended as the GUI project/session format: reload
        the file later to recover the analysis without rerunning everything.
        """

        h5py = _require_h5py()
        out = self.output_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as f:
            f.attrs["schema"] = "fibphot_gui_session"
            f.attrs["schema_version"] = 1
            f.attrs["metadata_json"] = _json_dumps(
                {
                    "source_path": self.source_path,
                    "output_dir": self.output_dir,
                    "history_policy": self.history_policy,
                    "pipeline": [c.to_dict() for c in self.pipeline],
                    "analyses": [c.to_dict() for c in self.analyses],
                    "batch_source_dir": self.batch_source_dir,
                    "batch_errors": self.batch_errors,
                    "errors": self.errors,
                }
            )
            if include_raw and self.raw_state is not None:
                _write_state_group(
                    f.create_group("raw_state"),
                    self.raw_state,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            if self.current_state is not None:
                _write_state_group(
                    f.create_group("current_state"),
                    self.current_state,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            if self.results:
                _write_results_group(
                    f.create_group("results"),
                    self.results,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            if include_batch:
                if self.raw_collection is not None:
                    g = f.create_group("raw_collection")
                    for i, state in enumerate(self.raw_collection.states):
                        _write_state_group(
                            g.create_group(f"state_{i:04d}"),
                            state,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                if self.current_collection is not None:
                    g = f.create_group("current_collection")
                    for i, state in enumerate(self.current_collection.states):
                        _write_state_group(
                            g.create_group(f"state_{i:04d}"),
                            state,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                if self.batch_report is not None:
                    g = f.create_group("batch_report")
                    g.attrs["failed_json"] = _json_dumps(
                        [f.__dict__ for f in self.batch_report.failed]
                    )
                    for i, report in enumerate(self.batch_report.reports):
                        rg = g.create_group(f"report_{i:04d}")
                        _write_state_group(
                            rg.create_group("state"),
                            report.state,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                        _write_results_group(
                            rg.create_group("results"),
                            report.results,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
        return out

    def load_session_h5(self, path: str | Path) -> Path:
        """Load a saved GUI session HDF5 file."""

        h5py = _require_h5py()
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        from ..collection import BatchFailure, BatchReport, PhotometryCollection

        with h5py.File(p, "r") as f:
            schema = _decode_str(f.attrs.get("schema", ""))
            if schema != "fibphot_gui_session":
                raise ValueError(
                    f"Not a fibphot GUI session file: schema={schema!r}"
                )
            payload = _json_loads(f.attrs.get("metadata_json", "{}"))
            if not isinstance(payload, dict):
                payload = {}
            self.source_path = payload.get("source_path")
            self.output_dir = str(payload.get("output_dir") or p.parent)
            self.history_policy = payload.get(
                "history_policy", self.history_policy
            )
            self.pipeline = [
                StageConfig.from_dict(x) for x in payload.get("pipeline", [])
            ]
            self.analyses = [
                AnalysisConfig.from_dict(x) for x in payload.get("analyses", [])
            ]
            self.batch_source_dir = payload.get("batch_source_dir")
            self.batch_errors = list(payload.get("batch_errors", []) or [])
            self.errors = list(payload.get("errors", []) or [])
            self.raw_state = (
                _read_state_group(f["raw_state"]) if "raw_state" in f else None
            )
            self.current_state = (
                _read_state_group(f["current_state"])
                if "current_state" in f
                else self.raw_state
            )
            self.results = (
                _read_results_group(f["results"]) if "results" in f else []
            )
            self.raw_collection = None
            if "raw_collection" in f:
                states = [
                    _read_state_group(f["raw_collection"][k])
                    for k in sorted(f["raw_collection"].keys())
                ]
                self.raw_collection = PhotometryCollection.from_iterable(states)
            self.current_collection = None
            if "current_collection" in f:
                states = [
                    _read_state_group(f["current_collection"][k])
                    for k in sorted(f["current_collection"].keys())
                ]
                self.current_collection = PhotometryCollection.from_iterable(
                    states
                )
            self.batch_report = None
            if "batch_report" in f:
                bg = f["batch_report"]
                reports = []
                for key in sorted(
                    k for k in bg.keys() if str(k).startswith("report_")
                ):
                    rg = bg[key]
                    st = _read_state_group(rg["state"])
                    results = (
                        tuple(_read_results_group(rg["results"]))
                        if "results" in rg
                        else ()
                    )
                    reports.append(PhotometryReport(st, results))
                failed_payload = _json_loads(bg.attrs.get("failed_json", "[]"))
                failed = (
                    tuple(BatchFailure(**x) for x in failed_payload)
                    if isinstance(failed_payload, list)
                    else ()
                )
                self.batch_report = BatchReport(tuple(reports), failed)
            self._undo_stack = []
            self._redo_stack = []
        return p

    def save_pipeline_json(self, path: str | Path) -> Path:
        payload = {
            "source_path": self.source_path,
            "history_policy": self.history_policy,
            "output_dir": self.output_dir,
            "pipeline": [c.to_dict() for c in self.pipeline],
            "analyses": [c.to_dict() for c in self.analyses],
        }
        out = self.output_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def load_pipeline_json(
        self, path: str | Path, *, recompute: bool = True
    ) -> None:
        payload = json.loads(self.output_path(path).read_text(encoding="utf-8"))
        self.pipeline = [
            StageConfig.from_dict(x) for x in payload.get("pipeline", [])
        ]
        self.analyses = [
            AnalysisConfig.from_dict(x) for x in payload.get("analyses", [])
        ]
        self.history_policy = payload.get("history_policy", self.history_policy)
        if recompute and self.raw_state is not None:
            self.recompute()
            self.rerun_analyses()

    def export(
        self, directory: str | Path, *, include_state: bool = True
    ) -> dict[str, Path]:
        directory = self.output_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        if self.current_state is None:
            raise RuntimeError("No state is loaded.")
        if include_state:
            p = directory / "processed_state.h5"
            save_state_h5(self.current_state, p)
            written["processed_state"] = p
        metrics = self.metrics_dataframe()
        if not metrics.empty:
            p = directory / "analysis_metrics.csv"
            metrics.to_csv(p, index=False)
            written["analysis_metrics"] = p
        arrays = self.arrays_dataframe()
        if not arrays.empty:
            p = directory / "analysis_arrays.csv"
            arrays.to_csv(p, index=False)
            written["analysis_arrays"] = p
        p = directory / "pipeline.json"
        self.save_pipeline_json(p)
        written["pipeline"] = p
        return written

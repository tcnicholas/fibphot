from __future__ import annotations

from .auc import AUC
from .aggregate import TraceStatistics, trace_statistics_from_aligned
from .connectivity import (
    CrossCorrelationAnalysis,
    GrangerCausalityAnalysis,
    PeakAlignedConnectivityAnalysis,
    cross_correlation,
    granger_pvalues,
)
from .events import (
    EventAlignedTraces,
    align_to_events,
    event_aligned_summary_dataframe,
    event_aligned_traces_dataframe,
    select_events_by_spacing,
)
from .peak_aligned import (
    PeakTriggeredAverage,
    peak_triggered_summary_dataframe,
    peak_triggered_traces_dataframe,
    result_to_event_aligned,
)
from .peaks import (
    PeakAnalysis,
    PeakEvent,
    PeakFit,
    PeaksByTemplate,
    biexponential_peak,
    build_template,
    matched_filter_trace,
    peak_result_to_dataframe,
    plot_peaks_by_template_result,
)
from .qc import QCAnalysis
from .report import AnalysisResult, AnalysisWindow, PhotometryReport
from .statistics import grouped_numeric_summary, nan_sem, nan_stats

__all__ = [
    "AUC",
    "TraceStatistics",
    "AnalysisResult",
    "AnalysisWindow",
    "CrossCorrelationAnalysis",
    "EventAlignedTraces",
    "GrangerCausalityAnalysis",
    "PeakAlignedConnectivityAnalysis",
    "PeakAnalysis",
    "PeakEvent",
    "PeakFit",
    "PeakTriggeredAverage",
    "PeaksByTemplate",
    "PhotometryReport",
    "QCAnalysis",
    "align_to_events",
    "biexponential_peak",
    "build_template",
    "cross_correlation",
    "event_aligned_summary_dataframe",
    "event_aligned_traces_dataframe",
    "granger_pvalues",
    "grouped_numeric_summary",
    "matched_filter_trace",
    "nan_sem",
    "nan_stats",
    "peak_result_to_dataframe",
    "peak_triggered_summary_dataframe",
    "peak_triggered_traces_dataframe",
    "plot_peaks_by_template_result",
    "result_to_event_aligned",
    "select_events_by_spacing",
    "trace_statistics_from_aligned",
]

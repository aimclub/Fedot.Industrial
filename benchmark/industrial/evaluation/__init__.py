"""Metric, report, and publication-pack entry points for benchmark evaluation."""

from benchmark.industrial.classification import compute_classification_metric, render_tsc_publication_pack
from benchmark.industrial.errors import BenchmarkClassificationError, BenchmarkRegressionError
from benchmark.industrial.evaluation.analytics import (
    SeriesComparisonResult,
    compare_models_on_series,
    render_publication_pack,
)
from benchmark.industrial.evaluation.okhs_quality import (
    OKHSSmoothingAcceptanceCriteria,
    OKHSSmoothingAcceptanceReport,
    evaluate_okhs_smoothing_acceptance,
    render_okhs_smoothing_acceptance_pack,
)
from benchmark.industrial.regression import compute_regression_metric, render_tser_publication_pack

__all__ = [
    "BenchmarkClassificationError",
    "BenchmarkRegressionError",
    "OKHSSmoothingAcceptanceCriteria",
    "OKHSSmoothingAcceptanceReport",
    "SeriesComparisonResult",
    "compare_models_on_series",
    "compute_classification_metric",
    "compute_regression_metric",
    "evaluate_okhs_smoothing_acceptance",
    "render_okhs_smoothing_acceptance_pack",
    "render_publication_pack",
    "render_tsc_publication_pack",
    "render_tser_publication_pack",
]

"""Metric, report, and publication-pack entry points for benchmark evaluation."""

_EXPORTS = {
    "BenchmarkClassificationError": "benchmark.industrial.errors",
    "BenchmarkRegressionError": "benchmark.industrial.errors",
    "OKHSSmoothingAcceptanceCriteria": "benchmark.industrial.evaluation.okhs_quality",
    "OKHSSmoothingAcceptanceReport": "benchmark.industrial.evaluation.okhs_quality",
    "SeriesComparisonResult": "benchmark.industrial.evaluation.analytics",
    "compare_models_on_series": "benchmark.industrial.evaluation.analytics",
    "compute_classification_metric": "benchmark.industrial.classification",
    "compute_regression_metric": "benchmark.industrial.regression",
    "evaluate_okhs_smoothing_acceptance": "benchmark.industrial.evaluation.okhs_quality",
    "render_okhs_smoothing_acceptance_pack": "benchmark.industrial.evaluation.okhs_quality",
    "render_publication_pack": "benchmark.industrial.evaluation.analytics",
    "render_tsc_publication_pack": "benchmark.industrial.classification",
    "render_tser_publication_pack": "benchmark.industrial.regression",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

"""Visualization helpers for benchmark artifacts and progress records."""

_EXPORTS = {
    "ForecastingProgressItemPayload": "benchmark.industrial.visualization.forecasting",
    "ForecastingProgressItemsVisualizer": "benchmark.industrial.visualization.forecasting",
    "ForecastingProgressVisualizationResult": "benchmark.industrial.visualization.forecasting",
    "build_forecast_comparison_frame": "benchmark.industrial.visualization.forecast_comparison",
    "build_forecast_comparison_from_progress_items": "benchmark.industrial.visualization.forecast_comparison",
    "build_forecast_metric_frame": "benchmark.industrial.visualization.forecast_comparison",
    "build_relative_gain_frame": "benchmark.industrial.visualization.forecasting",
    "file_md5": "benchmark.industrial.visualization.forecast_comparison",
    "render_benchmark_result_analysis_pack": "benchmark.industrial.visualization.benchmark_results",
    "render_forecast_comparison_pack": "benchmark.industrial.visualization.forecast_comparison",
    "visualize_forecasting_progress_items": "benchmark.industrial.visualization.forecasting",
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

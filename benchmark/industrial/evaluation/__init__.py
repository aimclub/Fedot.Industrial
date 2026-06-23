"""Metric, report, and publication-pack entry points for benchmark evaluation."""

_EXPORTS = {
    "AggregationInputContract": "benchmark.industrial.evaluation.aggregation",
    "AggregationOutputContract": "benchmark.industrial.evaluation.aggregation",
    "BenchmarkAggregationTables": "benchmark.industrial.evaluation.aggregation",
    "BenchmarkArtifactFrames": "benchmark.industrial.evaluation.aggregation",
    "BenchmarkClassificationError": "benchmark.industrial.errors",
    "BenchmarkRegressionError": "benchmark.industrial.errors",
    "OKHSSmoothingAcceptanceCriteria": "benchmark.industrial.evaluation.okhs_quality",
    "OKHSSmoothingAcceptanceReport": "benchmark.industrial.evaluation.okhs_quality",
    "SeriesComparisonResult": "benchmark.industrial.evaluation.analytics",
    "TaskAggregationRule": "benchmark.industrial.evaluation.aggregation",
    "build_benchmark_aggregate_tables": "benchmark.industrial.evaluation.aggregation",
    "build_leaderboard_frame": "benchmark.industrial.evaluation.aggregation",
    "compare_models_on_series": "benchmark.industrial.evaluation.analytics",
    "compute_classification_metric": "benchmark.industrial.classification",
    "compute_regression_metric": "benchmark.industrial.regression",
    "evaluate_okhs_smoothing_acceptance": "benchmark.industrial.evaluation.okhs_quality",
    "ResultAnalysisSpec": "benchmark.industrial.evaluation.result_analysis",
    "build_best_per_dataset_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_coverage_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_dataset_delta_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_dataset_difficulty_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_evolution_dynamics_frame": "benchmark.industrial.evaluation.evolution",
    "build_evolution_coverage_frame": "benchmark.industrial.evaluation.evolution",
    "build_generator_usage_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_mean_rank_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_operation_frequency_frame": "benchmark.industrial.evaluation.evolution",
    "build_model_diagnostics_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_parameter_metric_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_source_delta_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_pipeline_complexity_frame": "benchmark.industrial.evaluation.evolution",
    "build_status_summary_frame": "benchmark.industrial.evaluation.result_analysis",
    "build_topk_summary_frame": "benchmark.industrial.evaluation.result_analysis",
    "infer_metric_direction": "benchmark.industrial.evaluation.result_analysis",
    "load_benchmark_artifact_frames": "benchmark.industrial.evaluation.aggregation",
    "load_aggregate_metric_records": "benchmark.industrial.evaluation.result_analysis",
    "load_composition_history": "benchmark.industrial.evaluation.evolution",
    "load_incremental_kernel_diagnostics": "benchmark.industrial.evaluation.result_analysis",
    "load_incremental_metric_records": "benchmark.industrial.evaluation.result_analysis",
    "load_incremental_run_records": "benchmark.industrial.evaluation.result_analysis",
    "load_jsonl_table": "benchmark.industrial.evaluation.result_analysis",
    "load_result_table": "benchmark.industrial.evaluation.result_analysis",
    "load_result_sources": "benchmark.industrial.evaluation.result_analysis",
    "normalize_result_table": "benchmark.industrial.evaluation.result_analysis",
    "render_benchmark_aggregate_artifacts": "benchmark.industrial.evaluation.aggregation",
    "render_okhs_smoothing_acceptance_pack": "benchmark.industrial.evaluation.okhs_quality",
    "render_publication_pack": "benchmark.industrial.evaluation.analytics",
    "resolve_task_aggregation_rule": "benchmark.industrial.evaluation.aggregation",
    "render_evolution_analysis_pack": "benchmark.industrial.evaluation.evolution",
    "render_tsc_publication_pack": "benchmark.industrial.classification",
    "render_tser_publication_pack": "benchmark.industrial.regression",
    "select_notable_pipelines": "benchmark.industrial.evaluation.evolution",
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

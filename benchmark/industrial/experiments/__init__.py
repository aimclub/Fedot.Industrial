"""Suite configuration, manifest, preset, and registry helpers."""

_EXPORTS = {
    "run_forecasting_benchmark_suite": "benchmark.industrial.api",
    "run_tsc_benchmark_suite": "benchmark.industrial.api",
    "run_tser_benchmark_suite": "benchmark.industrial.api",
    "ArtifactSpec": "benchmark.industrial.core",
    "BenchmarkSuiteConfig": "benchmark.industrial.core",
    "DatasetSpec": "benchmark.industrial.core",
    "ModelSpec": "benchmark.industrial.core",
    "RunSpec": "benchmark.industrial.core",
    "TaskType": "benchmark.industrial.core",
    "BenchmarkManifestError": "benchmark.industrial.experiments.manifests",
    "build_suite_config_from_manifest": "benchmark.industrial.experiments.manifests",
    "load_manifest": "benchmark.industrial.experiments.manifests",
    "run_manifest": "benchmark.industrial.experiments.manifests",
    "run_manifest_path": "benchmark.industrial.experiments.manifests",
    "BenchmarkPresetError": "benchmark.industrial.experiments.presets",
    "load_preset_defaults": "benchmark.industrial.experiments.presets",
    "run_local_benchmark_preset": "benchmark.industrial.experiments.presets",
    "BenchmarkRunBundle": "benchmark.industrial.experiments.registry",
    "persist_run_bundle": "benchmark.industrial.experiments.registry",
    "run_registered_preset": "benchmark.industrial.experiments.registry",
    "run_registered_suite": "benchmark.industrial.experiments.registry",
    "DEFAULT_STAGE1_GENERATORS": "benchmark.industrial.experiments.kernel_learning",
    "DEFAULT_STAGE_METRICS": "benchmark.industrial.experiments.kernel_learning",
    "KernelLearningStage1ArtifactsLoader": "benchmark.industrial.experiments.kernel_learning",
    "KernelLearningStage1Runner": "benchmark.industrial.experiments.kernel_learning",
    "KernelLearningStage2Runner": "benchmark.industrial.experiments.kernel_learning",
    "build_stage2_initial_population": "benchmark.industrial.experiments.kernel_learning",
    "importance_report_from_selection": "benchmark.industrial.experiments.kernel_learning",
    "load_stage1_kernel_records": "benchmark.industrial.experiments.kernel_learning",
    "load_stage1_result_from_artifacts": "benchmark.industrial.experiments.kernel_learning",
    "resolve_existing_stage1_run_dir": "benchmark.industrial.experiments.kernel_learning",
    "run_stage2_for_dataset": "benchmark.industrial.experiments.kernel_learning",}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

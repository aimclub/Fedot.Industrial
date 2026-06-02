from __future__ import annotations

import sys
from pathlib import Path

from benchmark.v2 import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_forecasting_benchmark_suite,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_DATE = "140526"
M4_SUBSETS = ("monthly",)
M4_SAMPLE_SIZE = 5

DATASETS = tuple(
    DatasetSpec(
        benchmark="m4",
        dataset_name=f"m4_{subset.lower()}_kernel_learning",
        subset=subset,
        sample_size=M4_SAMPLE_SIZE,
        adapter_options={
            "use_local_files": True,
        },
    )
    for subset in M4_SUBSETS
)

KERNEL_LEARNING_FORECASTING_MODELS = (
    ModelSpec(
        adapter_name="naive_last_value",
        display_name="NaiveLastValue",
    ),
    ModelSpec(
        adapter_name="lagged_ridge_forecaster",
        display_name="LaggedRidgeForecaster",
        params={
            "window_size": 12,
            "stride": 1,
            "alpha": 1.0,
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_forecaster",
        display_name="KernelEnsembleForecaster_identity_shapelet",
        params={
            "generator_names": ("identity", "shapelet_extractor"),
            "kernel": "rbf",
            "gamma": "scale",
            "window_size": 12,
            "stride": 1,
            "head_type": "kernel_ridge",
            "alpha": 1.0,
            "selector_optimizer": "projected_gradient",
            "complexity_penalty": 0.01,
            "redundancy_penalty": 0.05,
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_forecaster",
        display_name="KernelEnsembleForecaster_embedding_nystrom_okhs",
        params={
            "generator_names": ("identity", "embedding_extractor"),
            "kernel": "rbf",
            "gamma": "scale",
            "kernel_approximation": "nystrom",
            "nystrom_components": 16,
            "window_size": 12,
            "stride": 1,
            "head_type": "okhs_kernel_ridge",
            "alpha": 1.0,
            "selector_optimizer": "projected_gradient",
        },
    ),
)

config = BenchmarkSuiteConfig(
    task_type=TaskType.FORECASTING,
    datasets=DATASETS,
    models=KERNEL_LEARNING_FORECASTING_MODELS,
    metrics=("mase", "smape", "owa", "rmse", "mae"),
    artifact_spec=ArtifactSpec(
        output_dir=f"benchmark/results/v2_kernel_learning/forecasting_suite_{EXPERIMENT_DATE}",
        persist_on_run=True,
    ),
    run_spec=RunSpec(
        run_name="kernel_learning_forecasting_suite",
        primary_metric="mae",
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
    ),
)

if __name__ == "__main__":
    result = run_forecasting_benchmark_suite(config)
    print(f"Run ID: {result.run_id}")
    print(f"Output dir: {result.config.artifact_spec.output_dir}")

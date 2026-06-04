from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.v2 import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_forecasting_benchmark_suite,
)
from benchmark.kernel_learning_experiment_controls import read_csv_env, read_positive_int_env

EXPERIMENT_DATE = "140526"
M4_SUBSETS = ("monthly",)
M4_SAMPLE_SIZE = 5
M4_SUBSET_ENV = "KERNEL_LEARNING_M4_SUBSETS"
M4_SAMPLE_SIZE_ENV = "KERNEL_LEARNING_M4_SAMPLE_SIZE"


def resolve_m4_subsets() -> tuple[str, ...]:
    return read_csv_env(M4_SUBSET_ENV) or M4_SUBSETS


def resolve_m4_sample_size() -> int:
    return read_positive_int_env(M4_SAMPLE_SIZE_ENV, M4_SAMPLE_SIZE) or M4_SAMPLE_SIZE

DATASETS = tuple(
    DatasetSpec(
        benchmark="m4",
        dataset_name=f"m4_{subset.lower()}_kernel_learning",
        subset=subset,
        sample_size=resolve_m4_sample_size(),
        adapter_options={
            "use_local_files": True,
        },
    )
    for subset in resolve_m4_subsets()
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
        resume_enabled=True,
    ),
)

if __name__ == "__main__":
    result = run_forecasting_benchmark_suite(config)
    print(f"Run ID: {result.run_id}")
    print(f"Output dir: {result.config.artifact_spec.output_dir}")

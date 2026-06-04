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
    run_tser_benchmark_suite,
)
from benchmark.kernel_learning_experiment_controls import apply_optional_limit, read_csv_env, read_positive_int_env

EXPERIMENT_DATE = "140526"
TSER_DATA_ROOT = PROJECT_ROOT / "fedot_ind" / "data"
TSER_DATASETS = (
    "NaturalGasPricesSentiment",
    "AppliancesEnergy",
    "ElectricityPredictor",
)
TSER_DATASET_ENV = "KERNEL_LEARNING_TSER_DATASETS"
TSER_DATASET_LIMIT_ENV = "KERNEL_LEARNING_TSER_LIMIT"


def resolve_tser_datasets() -> tuple[str, ...]:
    env_datasets = read_csv_env(TSER_DATASET_ENV)
    datasets = env_datasets or TSER_DATASETS
    return apply_optional_limit(datasets, read_positive_int_env(TSER_DATASET_LIMIT_ENV))

DATASETS = tuple(
    DatasetSpec(
        benchmark="local_tser",
        dataset_name=dataset_name,
        adapter_options={
            "local_data_root": str(TSER_DATA_ROOT),
            "download_if_missing": False,
        },
    )
    for dataset_name in resolve_tser_datasets()
)

KERNEL_LEARNING_MODELS = (
    ModelSpec(
        adapter_name="kernel_ensemble_regressor",
        display_name="KernelEnsembleRegressor_score_linear_summary",
        params={
            "generator_names": ("statistical_summary",),
            "kernel": "linear",
            "alpha": 1e-6,
            "selector_optimizer": "score",
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_regressor",
        display_name="KernelEnsembleRegressor_adaptive_rbf_summary",
        params={
            "generator_names": ("statistical_summary",),
            "kernel": "rbf",
            "gamma": "scale",
            "alpha": 1.0,
            "selector_optimizer": "projected_gradient",
            "complexity_penalty": 0.01,
            "redundancy_penalty": 0.05,
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_regressor",
        display_name="KernelEnsembleRegressor_shapelet_rbf",
        params={
            "generator_names": ("shapelet_extractor", "statistical_summary"),
            "kernel": "rbf",
            "gamma": "scale",
            "alpha": 1.0,
            "selector_optimizer": "projected_gradient",
            "complexity_penalty": 0.01,
            "redundancy_penalty": 0.05,
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_regressor",
        display_name="KernelEnsembleRegressor_embedding_nystrom",
        params={
            "generator_names": ("embedding_extractor", "statistical_summary"),
            "kernel": "rbf",
            "gamma": "scale",
            "kernel_approximation": "nystrom",
            "nystrom_components": 16,
            "alpha": 1.0,
            "selector_optimizer": "projected_gradient",
        },
    ),
)

config = BenchmarkSuiteConfig(
    task_type=TaskType.TS_REGRESSION,
    datasets=DATASETS,
    models=KERNEL_LEARNING_MODELS,
    metrics=("rmse", "mae", "r2"),
    artifact_spec=ArtifactSpec(
        output_dir=f"benchmark/results/v2_kernel_learning/tser_suite_{EXPERIMENT_DATE}",
        persist_on_run=True,
    ),
    run_spec=RunSpec(
        run_name="kernel_learning_tser_suite",
        primary_metric="rmse",
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
        resume_enabled=True,
    ),
)

if __name__ == "__main__":
    result = run_tser_benchmark_suite(config)
    print(f"Run ID: {result.run_id}")
    print(f"Output dir: {result.config.artifact_spec.output_dir}")

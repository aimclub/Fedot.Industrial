from __future__ import annotations
from benchmark.v2 import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    run_tser_benchmark_suite,
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPERIMENT_DATE = "140526"
TSER_DATA_ROOT = PROJECT_ROOT / "data"
TSER_DATASETS = (
    "NaturalGasPricesSentiment",
    "AppliancesEnergy",
    "ElectricityPredictor",
)

DATASETS = tuple(
    DatasetSpec(
        benchmark="local_tser",
        dataset_name=dataset_name,
        adapter_options={
            "local_data_root": str(TSER_DATA_ROOT),
            "download_if_missing": False,
        },
    )
    for dataset_name in TSER_DATASETS
)

KERNEL_LEARNING_MODELS = (
    ModelSpec(
        adapter_name="kernel_ensemble_regressor",
        display_name="KernelEnsembleRegressor_linear_summary",
        params={
            "generator_names": ("statistical_summary",),
            "kernel": "linear",
            "alpha": 1e-6,
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_regressor",
        display_name="KernelEnsembleRegressor_rbf_summary",
        params={
            "generator_names": ("statistical_summary",),
            "kernel": "rbf",
            "gamma": "scale",
            "alpha": 1.0,
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
    ),
)

if __name__ == "__main__":
    result = run_tser_benchmark_suite(config)
    print(f"Run ID: {result.run_id}")
    print(f"Output dir: {result.config.artifact_spec.output_dir}")

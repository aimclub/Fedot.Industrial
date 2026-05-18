from __future__ import annotations

import sys
from pathlib import Path

from fedot_ind.core.repository.constanst_repository import UNI_CLF_BENCH

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
    run_tsc_benchmark_suite,
)

EXPERIMENT_DATE = "140526"
UCR_DATA_ROOT = PROJECT_ROOT / "data"
UCR_DATASETS = ("Lightning7", "ECG200", "Coffee")
UCR_DATASETS = UNI_CLF_BENCH
NON_TOPOLOGICAL_GENERATORS = (
    "quantile_extractor_torch",
    "wavelet_extractor",
    "fourier_extractor",
    # "eigen_extractor",
    "recurrence_extractor",
)

DATASETS = tuple(
    DatasetSpec(
        benchmark="ucr",
        dataset_name=dataset_name,
        adapter_options={
            "local_data_root": str(UCR_DATA_ROOT),
            "download_if_missing": True,
        },
    )
    for dataset_name in UCR_DATASETS
)

KERNEL_LEARNING_MODELS = (
    # ModelSpec(
    #     adapter_name="kernel_ensemble_classifier",
    #     display_name="KernelEnsembleClassifier_linear_summary",
    #     params={
    #         "generator_names": ("statistical_summary",),
    #         "kernel": "linear",
    #         "C": 10.0,
    #     },
    # ),
    # ModelSpec(
    #     adapter_name="kernel_ensemble_classifier",
    #     display_name="KernelEnsembleClassifier_rbf_summary",
    #     params={
    #         "generator_names": ("statistical_summary",),
    #         "kernel": "rbf",
    #         "gamma": "scale",
    #         "C": 1.0,
    #     },
    # ),
    ModelSpec(
        adapter_name="kernel_ensemble_classifier",
        display_name="KernelEnsembleClassifier_all_non_topological",
        params={
            "generator_names": NON_TOPOLOGICAL_GENERATORS,
            "kernel": "rbf",
            "gamma": "scale",
            "C": 1.0,
            "complexity_penalty": 0.01,
            "redundancy_penalty": 0.05,
            "min_weight": 0.05,
            'importance_threshold': 0.15
        },
    ),
)

config = BenchmarkSuiteConfig(
    task_type=TaskType.TS_CLASSIFICATION,
    datasets=DATASETS,
    models=KERNEL_LEARNING_MODELS,
    metrics=("accuracy", "balanced_accuracy", "f1_macro"),
    artifact_spec=ArtifactSpec(
        output_dir=f"benchmark/results/v2_kernel_learning/ucr_suite_{EXPERIMENT_DATE}",
        persist_on_run=True,
    ),
    run_spec=RunSpec(
        run_name="kernel_learning_ucr_suite",
        primary_metric="f1",
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
    ),
)

if __name__ == "__main__":
    result = run_tsc_benchmark_suite(config)
    print(f"Run ID: {result.run_id}")
    print(f"Output dir: {result.config.artifact_spec.output_dir}")

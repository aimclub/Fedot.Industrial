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
    discover_local_ucr_datasets,
    run_tsc_benchmark_suite,
)
from benchmark.kernel_learning_experiment_controls import apply_optional_limit, read_csv_env, read_positive_int_env
from fedot_ind.core.repository.constanst_repository import UNI_CLF_BENCH

EXPERIMENT_DATE = "020626"
UCR_DATA_ROOT = PROJECT_ROOT / "data"
UCR_DATASETS = ()
UCR_DATASET_ENV = "KERNEL_LEARNING_UCR_DATASETS"
UCR_DATASET_LIMIT_ENV = "KERNEL_LEARNING_UCR_LIMIT"
NON_TOPOLOGICAL_GENERATORS = (
    "quantile_extractor_torch",
    "wavelet_extractor",
    "fourier_extractor",
    "recurrence_extractor",
    "tabular_extractor",
)


def resolve_ucr_datasets() -> tuple[str, ...]:
    env_datasets = read_csv_env(UCR_DATASET_ENV)
    if env_datasets:
        datasets = env_datasets
    elif UCR_DATASETS:
        datasets = tuple(UCR_DATASETS)
    else:
        datasets = discover_local_ucr_datasets(UCR_DATA_ROOT, allowed_names=UNI_CLF_BENCH)
    return apply_optional_limit(datasets, read_positive_int_env(UCR_DATASET_LIMIT_ENV))


DATASETS = tuple(
    DatasetSpec(
        benchmark="ucr",
        dataset_name=dataset_name,
        adapter_options={
            "local_data_root": str(UCR_DATA_ROOT),
            "download_if_missing": True,
        },
    )
    for dataset_name in resolve_ucr_datasets()
)

KERNEL_LEARNING_MODELS = (
    ModelSpec(
        adapter_name="kernel_ensemble_classifier",
        display_name="KernelEnsembleClassifier_score_baseline_summary",
        params={
            "generator_names": ("statistical_summary",),
            "kernel": "linear",
            "C": 10.0,
            "selector_optimizer": "score",
            "importance_threshold": 0.05,
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_classifier",
        display_name="KernelEnsembleClassifier_adaptive_all_non_topological",
        params={
            "generator_names": NON_TOPOLOGICAL_GENERATORS,
            "kernel": "rbf",
            "gamma": "scale",
            "torch_device": "auto",
            "C": 1.0,
            "selector_optimizer": "projected_gradient",
            "complexity_penalty": 0.01,
            "redundancy_penalty": 0.05,
            "min_weight": 0.05,
            "importance_threshold": 0.05,
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_classifier",
        display_name="KernelEnsembleClassifier_shapelet_motif_rbf",
        params={
            "generator_names": ("shapelet_extractor", "statistical_summary"),
            "kernel": "rbf",
            "gamma": "scale",
            "C": 1.0,
            "selector_optimizer": "projected_gradient",
            "complexity_penalty": 0.01,
            "redundancy_penalty": 0.05,
            "importance_threshold": 0.05,
        },
    ),
    ModelSpec(
        adapter_name="kernel_ensemble_classifier",
        display_name="KernelEnsembleClassifier_embedding_nystrom",
        params={
            "generator_names": ("embedding_extractor", "statistical_summary"),
            "kernel": "rbf",
            "gamma": "scale",
            "kernel_approximation": "nystrom",
            "nystrom_components": 16,
            "C": 1.0,
            "selector_optimizer": "projected_gradient",
            "importance_threshold": 0.05,
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
        primary_metric="f1_macro",
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
        resume_enabled=True,
    ),
)

if __name__ == "__main__":
    result = run_tsc_benchmark_suite(config)
    print(f"Run ID: {result.run_id}")
    print(f"Output dir: {result.config.artifact_spec.output_dir}")

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.v2 import (  # noqa: E402
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    discover_local_supervised_datasets,
    run_tsc_benchmark_suite,
)
from benchmark.v2.local_io import LocalDatasetParseError, load_local_supervised_split  # noqa: E402
from fedot_ind.core.repository.constanst_repository import MULTI_CLF_BENCH, UNI_CLF_BENCH  # noqa: E402

EXPERIMENT_DATE = "190526"
UCR_DATA_ROOT = PROJECT_ROOT / "data"
PDL_DATASETS = ()

MIN_CLASSES = 3
MAX_TRAIN_SAMPLES = 100
MAX_MEDIAN_SAMPLES_PER_CLASS = 40

ALLOWED_DATASETS = tuple(dict.fromkeys(tuple(UNI_CLF_BENCH) + tuple(MULTI_CLF_BENCH)))
# ALLOWED_DATASETS = tuple(dict.fromkeys(tuple(MULTI_CLF_BENCH)))
PDL_DEFAULT_PARAMS = {
    "model": "rf",
    "backend": "auto",
    "pairing_policy": "adaptive_anchors",
    "max_pairs": 250_000,
    "anchors_per_class": 20,
    "pair_feature_mode": "concat_diff",
    "chunk_size": 8192,
    "random_state": 42,
    "n_estimators": 100,
    "n_jobs": -1,
}

QUANTILE_PARAMS = {
    "window_size": 10,
    "stride": 1,
    "add_global_features": True,
}


def resolve_small_multiclass_datasets() -> tuple[str, ...]:
    if PDL_DATASETS:
        return tuple(PDL_DATASETS)
    discovered = discover_local_supervised_datasets(UCR_DATA_ROOT, allowed_names=ALLOWED_DATASETS)
    return tuple(name for name in discovered if _passes_small_multiclass_filter(name))


def _passes_small_multiclass_filter(dataset_name: str) -> bool:
    try:
        split = load_local_supervised_split(dataset_name, data_root=UCR_DATA_ROOT)
    except LocalDatasetParseError:
        return False
    target = np.asarray(split.train_target).reshape(-1)
    labels, counts = np.unique(target, return_counts=True)
    if len(labels) < MIN_CLASSES:
        return False
    if len(target) > MAX_TRAIN_SAMPLES:
        return False
    return float(np.median(counts)) <= MAX_MEDIAN_SAMPLES_PER_CLASS


DATASETS = tuple(
    DatasetSpec(
        benchmark="ucr",
        dataset_name=dataset_name,
        adapter_options={
            "local_data_root": str(UCR_DATA_ROOT),
            "download_if_missing": True,
        },
    )
    for dataset_name in resolve_small_multiclass_datasets()
)

MODELS = (
    ModelSpec(
        adapter_name="quantile_rf_classifier",
        display_name="QuantileExtractorTorch_RF",
        params={
            "quantile_params": QUANTILE_PARAMS,
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    ModelSpec(
        adapter_name="pdl_classifier",
        display_name="PDL_RF_raw",
        params=PDL_DEFAULT_PARAMS,
    ),
    ModelSpec(
        adapter_name="pdl_quantile_classifier",
        display_name="PDL_RF_quantile",
        params={
            **PDL_DEFAULT_PARAMS,
            "quantile_params": QUANTILE_PARAMS,
        },
    ),
)

config = BenchmarkSuiteConfig(
    task_type=TaskType.TS_CLASSIFICATION,
    datasets=DATASETS,
    models=MODELS,
    metrics=("accuracy", "balanced_accuracy", "f1_macro"),
    artifact_spec=ArtifactSpec(
        output_dir=f"benchmark/results/v2_pdl/small_multiclass_ucr_{EXPERIMENT_DATE}",
        persist_on_run=True,
    ),
    run_spec=RunSpec(
        run_name="pdl_small_multiclass_ucr",
        primary_metric="f1_macro",
        show_progress=True,
        progress_leave=False,
        progress_log_errors=True,
        progress_log_summaries=True,
    ),
)

if __name__ == "__main__":
    result = run_tsc_benchmark_suite(config)
    print(f"Run ID: {result.run_id}")
    print(f"Datasets: {len(result.config.datasets)}")
    print(f"Output dir: {result.config.artifact_spec.output_dir}")

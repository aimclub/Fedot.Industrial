from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.experiments.kernel_learning.configs import (
    KernelLearningM4ExperimentConfig,
    KernelLearningTSERExperimentConfig,
    KernelLearningTwoStageUCRExperimentConfig,
    KernelLearningUCRExperimentConfig,
    run_kernel_learning_suite,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[3]
DEFAULTS_PATH = PACKAGE_ROOT / "full_run_defaults.json"
DEFAULTS_VERSION = "industrial_real_world_full_runs@1"
TSER_REFERENCE_TABLE = PROJECT_ROOT / "benchmark" / "results" / "time_series_multi_reg_comparasion_2024.csv"


@lru_cache(maxsize=1)
def load_full_run_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    defaults_path = Path(path)
    payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Full-run defaults root must be a mapping: {defaults_path}")
    version = str(payload.get("version", ""))
    if version != DEFAULTS_VERSION:
        raise ValueError(f"Unsupported full-run defaults version: {version}")
    return payload


def available_full_run_names() -> tuple[str, ...]:
    return tuple(sorted(load_full_run_defaults()["runs"]))


def build_full_run_config(name: str):
    payload = _run_payload(name)
    experiment = str(payload["experiment"])
    output_dir = PROJECT_ROOT / str(payload["output_dir"]) if payload.get("output_dir") else None
    if experiment == "kernel_learning_ucr":
        return KernelLearningUCRExperimentConfig(
            output_dir=output_dir,
            persist_on_run=bool(payload.get("persist_on_run", True)),
            run_name=str(payload.get("run_name", "kernel_learning_ucr_full")),
        )
    if experiment == "kernel_learning_tser":
        datasets = payload.get("datasets")
        if datasets is None:
            datasets = _discover_tser_datasets()
        return KernelLearningTSERExperimentConfig(
            output_dir=output_dir,
            persist_on_run=bool(payload.get("persist_on_run", True)),
            run_name=str(payload.get("run_name", "kernel_learning_tser_full")),
            datasets=tuple(str(item) for item in datasets),
        )
    if experiment == "kernel_learning_m4":
        sample_size = payload.get("sample_size")
        return KernelLearningM4ExperimentConfig(
            output_dir=output_dir,
            persist_on_run=bool(payload.get("persist_on_run", True)),
            run_name=str(payload.get("run_name", "kernel_learning_m4_full")),
            subsets=tuple(str(item) for item in payload.get("subsets", ("Monthly",))),
            sample_size=int(sample_size) if sample_size is not None else None,
        )
    if experiment == "kernel_learning_two_stage_ucr":
        return KernelLearningTwoStageUCRExperimentConfig(
            stage1_output_dir=PROJECT_ROOT / str(payload["stage1_output_dir"]),
            stage2_output_dir=PROJECT_ROOT / str(payload["stage2_output_dir"]),
            run_stage1=bool(payload.get("run_stage1", False)),
        )
    raise ValueError(f"Unsupported full-run experiment: {experiment}")


def run_full_benchmark(name: str):
    config = build_full_run_config(name)
    if isinstance(config, KernelLearningTwoStageUCRExperimentConfig):
        stage1_result = config.load_or_run_stage1()
        return config.run_stage2(stage1_result)
    return run_kernel_learning_suite(config)


def _run_payload(name: str) -> dict[str, Any]:
    runs = load_full_run_defaults()["runs"]
    if name not in runs:
        raise ValueError(f"Unknown full-run name: {name}. Known: {', '.join(sorted(runs))}")
    return dict(runs[name])


def _discover_tser_datasets(data_root: Path | None = None) -> tuple[str, ...]:
    root = data_root or (PROJECT_ROOT / "fedot_ind" / "data")
    if not root.exists():
        return KernelLearningTSERExperimentConfig.DEFAULT_TSER_DATASETS
    candidates = _load_tser_reference_dataset_names()
    datasets = tuple(name for name in candidates if _has_local_split(root, name))
    return datasets or KernelLearningTSERExperimentConfig.DEFAULT_TSER_DATASETS


def _load_tser_reference_dataset_names(path: Path = TSER_REFERENCE_TABLE) -> tuple[str, ...]:
    if not path.exists():
        return KernelLearningTSERExperimentConfig.DEFAULT_TSER_DATASETS
    frame = pd.read_csv(path, sep=None, engine="python")
    if frame.empty:
        return KernelLearningTSERExperimentConfig.DEFAULT_TSER_DATASETS
    dataset_column = frame.columns[0]
    names = tuple(str(value) for value in frame[dataset_column].dropna().unique())
    return names or KernelLearningTSERExperimentConfig.DEFAULT_TSER_DATASETS


def _has_local_split(root: Path, dataset_name: str) -> bool:
    dataset_dir = root / dataset_name
    if not dataset_dir.is_dir():
        return False
    train_base = dataset_dir / f"{dataset_name}_TRAIN"
    test_base = dataset_dir / f"{dataset_name}_TEST"
    return any(train_base.with_suffix(extension).exists() and test_base.with_suffix(extension).exists()
               for extension in (".tsv", ".csv", ".ts"))


if __name__ == "__main__":
    print(available_full_run_names())

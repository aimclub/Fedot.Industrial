from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmark.industrial import BenchmarkSuiteConfig
from examples.utils.current_api import (
    build_forecasting_suite_config,
    build_tsc_suite_config,
    build_tser_suite_config,
    run_tsc_example,
    run_tser_example,
)

EXAMPLE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_ROOT.parents[1]
DEFAULTS_PATH = EXAMPLE_ROOT / "real_world_defaults.json"
DEFAULTS_VERSION = "industrial_real_world_examples@1"
EXTERNAL_DATA_MANIFEST_PATH = EXAMPLE_ROOT / "external_data_manifest.json"
EXTERNAL_DATA_MANIFEST_VERSION = "industrial_real_world_external_data@1"


@lru_cache(maxsize=1)
def load_real_world_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    defaults_path = Path(path)
    payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Real-world example defaults root must be a mapping: {defaults_path}")
    version = str(payload.get("version", ""))
    if version != DEFAULTS_VERSION:
        raise ValueError(f"Unsupported real-world example defaults version: {version}")
    return payload


@lru_cache(maxsize=1)
def load_external_data_manifest(path: str | Path = EXTERNAL_DATA_MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"External data manifest root must be a mapping: {manifest_path}")
    version = str(payload.get("version", ""))
    if version != EXTERNAL_DATA_MANIFEST_VERSION:
        raise ValueError(f"Unsupported external data manifest version: {version}")
    return payload


def build_classification_benchmark_config(output_dir: str | Path | None = None) -> BenchmarkSuiteConfig:
    return build_tsc_suite_config(output_dir=output_dir)


def run_classification_benchmark(output_dir: str | Path | None = None):
    return run_tsc_example(output_dir=output_dir)


def build_regression_benchmark_config(output_dir: str | Path | None = None) -> BenchmarkSuiteConfig:
    return build_tser_suite_config(output_dir=output_dir)


def run_regression_benchmark(output_dir: str | Path | None = None):
    return run_tser_example(output_dir=output_dir)


def build_forecasting_benchmark_config(output_dir: str | Path | None = None) -> BenchmarkSuiteConfig:
    return build_forecasting_suite_config(output_dir=output_dir)


def skab_context(folder: str = "valve1") -> dict[str, Any]:
    root = PROJECT_ROOT / "examples" / "utils" / "data" / "anomaly_detection" / "skab"
    folder_dir = root / folder
    datasets = tuple(path.stem for path in sorted(folder_dir.glob("*.csv"))) if folder_dir.exists() else ()
    return {
        "task_type": "anomaly_detection",
        "data_root": str(root),
        "folder": folder,
        "datasets": datasets,
        "train_data_size": "anomaly-free",
    }


def kaggle_forecasting_context(data_dir: str | Path = "examples/utils/data/forecasting/kaggle_inventory") -> dict[str, Any]:
    defaults = load_real_world_defaults()["kaggle_forecasting"]
    data_root = Path(data_dir)
    return {
        "task_type": "forecasting",
        "data_root": str(data_root),
        "train_path": str(data_root / defaults["train_file"]),
        "test_path": str(data_root / defaults["test_file"]),
        "submission_path": str(data_root / defaults["submission_file"]),
        "warehouse_count": len(defaults["warehouses"]),
    }


def eeg_classification_context() -> dict[str, Any]:
    defaults = load_real_world_defaults()["eeg_classification"]
    data_root = PROJECT_ROOT / defaults["data_root"]
    return {
        "task_type": "ts_classification",
        "data_root": str(data_root),
        "feature_file": str(data_root / defaults["feature_file"]),
        "target_file": str(data_root / defaults["target_file"]),
        "notebook": str(
            PROJECT_ROOT
            / "examples/real_world_examples/industrial_examples/eeg/classification/"
            / "harmful_brain_activity_classification.ipynb"
        ),
    }


def debet_forecasting_context() -> dict[str, Any]:
    defaults = load_real_world_defaults()["debet_forecasting"]
    data_root = PROJECT_ROOT / defaults["data_root"]
    return {
        "task_type": "forecasting",
        "data_root": str(data_root),
        "config_path": str(data_root / defaults["config_file"]),
        "supported_models": tuple(defaults["supported_models"]),
        "local_output_dirs": tuple(str(data_root / name) for name in defaults["output_dirs"]),
        "recommended_visualization": "benchmark.industrial.evaluation.render_publication_pack",
    }


def external_data_summary() -> dict[str, Any]:
    manifest = load_external_data_manifest()
    return {
        "version": manifest["version"],
        "delivery_mode": manifest["delivery"]["mode"],
        "dvc_remote": manifest["delivery"]["dvc_remote_name"],
        "source_keys": tuple(source["key"] for source in manifest["sources"]),
    }


def real_world_preflight_summary() -> dict[str, Any]:
    from examples.real_world_examples.benchmark_example.analysis_of_results import (
        available_analysis_names,
    )
    from examples.real_world_examples.industrial_examples import list_domain_scenarios

    return {
        "defaults_version": load_real_world_defaults()["version"],
        "external_data": external_data_summary(),
        "analysis_notebooks": available_analysis_names(),
        "domain_scenarios": list_domain_scenarios(),
        "skab": skab_context("valve1"),
        "eeg": eeg_classification_context(),
        "debet": debet_forecasting_context(),
    }


def config_summary(config: BenchmarkSuiteConfig) -> dict[str, Any]:
    return {
        "task_type": config.task_type.value,
        "datasets": [dataset.dataset_name for dataset in config.datasets],
        "models": [model.display_name for model in config.models],
        "persist_on_run": config.artifact_spec.persist_on_run,
    }


def result_summary(result) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "task_type": result.config.task_type.value,
        "successful_runs": sum(1 for record in result.run_records if record.status.value == "success"),
        "primary_metric": result.aggregate_report.primary_metric,
    }


if __name__ == "__main__":
    print(real_world_preflight_summary())

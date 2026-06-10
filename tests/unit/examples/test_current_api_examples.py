from __future__ import annotations

from pathlib import Path

from benchmark.industrial import RunStatus, TaskType
from benchmark.industrial.experiments.manifests import load_manifest, render_resolved_manifest, validate_manifest
from examples.utils.current_api import (
    build_forecasting_suite_config,
    build_kernel_learning_ucr_config_preview,
    run_all_lightweight_examples,
    run_tsc_example,
    run_tser_example,
)


def test_current_api_tsc_example_runs_without_manual_setup(tmp_path: Path) -> None:
    result = run_tsc_example(tmp_path / "tsc")

    assert result.config.task_type is TaskType.TS_CLASSIFICATION
    assert result.config.artifact_spec.persist_on_run is False
    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name == "NearestCentroid" for record in result.run_records)


def test_current_api_tser_example_runs_without_manual_setup(tmp_path: Path) -> None:
    result = run_tser_example(tmp_path / "tser")

    assert result.config.task_type is TaskType.TS_REGRESSION
    assert result.config.artifact_spec.persist_on_run is False
    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name == "LinearRegressor" for record in result.run_records)


def test_current_api_forecasting_example_builds_config_without_optional_runtime(tmp_path: Path) -> None:
    config = build_forecasting_suite_config(tmp_path / "forecasting")

    assert config.task_type is TaskType.FORECASTING
    assert config.artifact_spec.persist_on_run is False
    assert [dataset.dataset_name for dataset in config.datasets] == ["toy_forecasting_current_api"]
    assert any(model.adapter_name == "naive_last_value" for model in config.models)


def test_current_api_examples_can_run_as_group(tmp_path: Path) -> None:
    results = run_all_lightweight_examples(tmp_path)

    assert set(results) == {"tsc", "tser"}
    assert all(result.config.artifact_spec.persist_on_run is False for result in results.values())
    assert all(any(record.status is RunStatus.SUCCESS for record in result.run_records) for result in results.values())


def test_kernel_learning_example_builds_current_ucr_config_without_running(tmp_path: Path) -> None:
    config = build_kernel_learning_ucr_config_preview(tmp_path / "kernel_learning", datasets=("Lightning7",))

    assert config.task_type is TaskType.TS_CLASSIFICATION
    assert config.artifact_spec.persist_on_run is False
    assert config.run_spec.run_name == "kernel_learning_ucr_preview"
    assert [dataset.dataset_name for dataset in config.datasets] == ["Lightning7"]
    assert any(model.adapter_name == "kernel_ensemble_classifier" for model in config.models)


def test_current_api_json_manifest_uses_current_manifest_version() -> None:
    payload = load_manifest("examples/utils/current_api/manifests/toy_tser_suite.json")
    resolved = render_resolved_manifest(payload)

    assert payload["version"] == "benchmark_industrial_manifest@1"
    assert resolved["task_type"] == "ts_regression"
    assert resolved["datasets"][0]["benchmark"] == "in_memory_tser"


def test_current_api_preset_manifest_uses_current_manifest_version() -> None:
    payload = load_manifest("examples/utils/current_api/manifests/m4_daily_preset.json")
    validated = validate_manifest(payload)

    assert validated["version"] == "benchmark_industrial_manifest@1"
    assert validated["kind"] == "preset"
    assert validated["preset_name"] == "m4"

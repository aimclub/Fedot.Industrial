from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmark.industrial import (
    TaskType,
    build_benchmark_aggregate_tables,
    build_leaderboard_frame,
    render_benchmark_aggregate_artifacts,
    resolve_task_aggregation_rule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_UCR_RUN = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
    / "v2_kernel_learning"
    / "ucr_suite_020626"
    / "kernel_learning_ucr_suite_c44f2a9f6c"
)
REFERENCE_TSER_RUN = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
    / "v2_kernel_learning"
    / "tser_suite_140526"
    / "kernel_learning_tser_suite_3d9e7761d0"
)
REFERENCE_M4_RUN = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
    / "v2_kernel_learning"
    / "forecasting_suite_140526"
    / "kernel_learning_forecasting_suite_f184092168"
)


def test_task_aggregation_rules_define_task_specific_defaults() -> None:
    classification = resolve_task_aggregation_rule("classification")
    regression = resolve_task_aggregation_rule(TaskType.TS_REGRESSION)
    forecasting = resolve_task_aggregation_rule("m4")

    assert classification.primary_metric == "accuracy"
    assert classification.metric_direction == "higher"
    assert classification.leaderboard_group_columns == ("dataset_name", "model_name")
    assert regression.primary_metric == "rmse"
    assert regression.metric_direction == "lower"
    assert forecasting.primary_metric == "mae"
    assert forecasting.count_column == "n_series"
    assert forecasting.leaderboard_group_columns == ("benchmark", "dataset_name", "model_name")


def test_incremental_records_build_reference_style_aggregate_outputs(tmp_path: Path) -> None:
    root = tmp_path / "run"
    records = root / "records"
    records.mkdir(parents=True)
    _append_jsonl(
        records / "runs.jsonl",
        _run_record("run_1", "D1", "ModelA", "success", {"accuracy": 0.9}),
    )
    _append_jsonl(
        records / "runs.jsonl",
        _run_record("run_1", "D1", "ModelB", "success", {"accuracy": 0.8}),
    )
    _append_jsonl(
        records / "runs.jsonl",
        _run_record("run_1", "D2", "ModelB", "failed", {}),
    )
    for row in (
        _metric_record("run_1", "D1", "ModelA", "accuracy", 0.9, "success"),
        _metric_record("run_1", "D1", "ModelB", "accuracy", 0.8, "success"),
        _metric_record("run_1", "D2", "ModelB", "accuracy", 0.7, "failed"),
        _metric_record("run_1", "D1", "ModelA", "f1_macro", 0.87, "success"),
    ):
        _append_jsonl(records / "metrics.jsonl", row)
    _append_jsonl(
        records / "predictions.jsonl",
        {
            "run_id": "run_1",
            "benchmark": "ucr",
            "dataset_name": "D1",
            "subset": "default",
            "model_name": "ModelA",
            "sample_index": 0,
            "y_true": "0",
            "y_pred": "0",
            "status": "success",
        },
    )
    _append_jsonl(
        records / "kernel_diagnostics.jsonl",
        {
            "run_id": "run_1",
            "dataset_name": "D1",
            "model_name": "ModelA",
            "summary": {"n_kernels": 1},
        },
    )

    tables = build_benchmark_aggregate_tables(root, task_type="classification")
    manifest = render_benchmark_aggregate_artifacts(root, output_dir=tmp_path / "aggregate", task_type="classification")

    assert tables.run_metadata["status_counts"] == {"failed": 1, "success": 2}
    assert tables.run_metadata["record_counts"]["kernel_diagnostics"] == 1
    assert tables.leaderboard["model_name"].tolist() == ["ModelA", "ModelB"]
    assert tables.leaderboard["accuracy"].tolist() == [0.9, 0.8]
    assert tables.leaderboard["n_runs"].tolist() == [1, 1]
    assert (tmp_path / "aggregate" / "leaderboard.csv").is_file()
    assert (tmp_path / "aggregate" / "run_metadata.json").is_file()
    assert (tmp_path / "aggregate" / "summary.md").is_file()
    assert any(Path(record.path).name == "artifact_manifest.json" for record in manifest)


def test_leaderboard_aggregation_is_order_invariant() -> None:
    rule = resolve_task_aggregation_rule("classification")
    rows = pd.DataFrame(
        [
            _metric_record("run_1", "D1", "ModelB", "accuracy", 0.8, "success"),
            _metric_record("run_1", "D1", "ModelA", "accuracy", 0.9, "success"),
            _metric_record("run_1", "D2", "ModelA", "accuracy", 0.7, "success"),
        ]
    )
    shuffled = rows.sample(frac=1.0, random_state=42).reset_index(drop=True)

    first = build_leaderboard_frame(rows, rule)
    second = build_leaderboard_frame(shuffled, rule)

    pd.testing.assert_frame_equal(first, second)


def test_forecasting_aggregation_uses_lower_metric_and_series_count(tmp_path: Path) -> None:
    records = tmp_path / "forecasting_run" / "records"
    records.mkdir(parents=True)
    for dataset_name, series_id, model_name, mae in (
        ("M100", "S1", "ForecasterA", 1.0),
        ("M100", "S2", "ForecasterA", 2.0),
        ("M100", "S1", "ForecasterB", 1.2),
        ("M100", "S2", "ForecasterB", 0.8),
    ):
        _append_jsonl(records / "runs.jsonl", _run_record("run_f", dataset_name,
                      model_name, "success", {"mae": mae}, series_id))
        _append_jsonl(
            records /
            "metrics.jsonl",
            _metric_record(
                "run_f",
                dataset_name,
                model_name,
                "mae",
                mae,
                "success",
                series_id))

    tables = build_benchmark_aggregate_tables(tmp_path / "forecasting_run", task_type="forecasting")

    assert tables.leaderboard["model_name"].tolist() == ["ForecasterB", "ForecasterA"]
    assert tables.leaderboard["mae"].tolist() == [1.0, 1.5]
    assert tables.leaderboard["n_series"].tolist() == [2, 2]


@pytest.mark.parametrize("run_path,task_type,primary_metric,expected_columns",
                         [(REFERENCE_UCR_RUN, "classification", "accuracy",
                           {"dataset_name", "model_name", "accuracy", "n_runs", "rank"}),
                          (REFERENCE_TSER_RUN, "regression", "rmse",
                           {"dataset_name", "model_name", "rmse", "n_runs", "rank"}),
                          (REFERENCE_M4_RUN, "forecasting", "mae",
                           {"benchmark", "dataset_name", "model_name", "mae", "n_series", "rank"}),],)
def test_v2_kernel_learning_reference_runs_can_be_reaggregated_when_local_artifacts_exist(
    run_path: Path,
    task_type: str,
    primary_metric: str,
    expected_columns: set[str],
) -> None:
    if not run_path.exists():
        pytest.skip(f"Local v2 Kernel Learning reference artifacts are not available: {run_path}")
    tables = build_benchmark_aggregate_tables(
        run_path,
        task_type=task_type,
        primary_metric=primary_metric,
    )

    assert tables.run_metadata["status_counts"].get("success", 0) > 0
    assert not tables.leaderboard.empty
    assert expected_columns.issubset(tables.leaderboard.columns)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")


def _run_record(
    run_id: str,
    dataset_name: str,
    model_name: str,
    status: str,
    metrics_summary: dict[str, float],
    series_id: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "benchmark": "ucr" if series_id is None else "m4",
        "dataset_name": dataset_name,
        "subset": "default",
        "series_id": series_id or dataset_name,
        "model_name": model_name,
        "status": status,
        "tags": ["industrial"],
        "message": "",
        "metrics_summary": metrics_summary,
        "metadata": {},
    }


def _metric_record(
    run_id: str,
    dataset_name: str,
    model_name: str,
    metric_name: str,
    metric_value: float,
    status: str,
    series_id: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "benchmark": "ucr" if series_id is None else "m4",
        "dataset_name": dataset_name,
        "subset": "default",
        "series_id": series_id or dataset_name,
        "model_name": model_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "status": status,
        "horizon_index": None,
    }

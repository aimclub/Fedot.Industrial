from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark.industrial import (
    ResultAnalysisSpec,
    build_coverage_frame,
    build_forecast_comparison_frame,
    build_forecast_comparison_from_progress_items,
    build_forecast_metric_frame,
    build_best_per_dataset_frame,
    build_dataset_delta_frame,
    build_generator_usage_frame,
    build_mean_rank_frame,
    build_model_diagnostics_frame,
    build_operation_frequency_frame,
    build_topk_summary_frame,
    file_md5,
    load_incremental_metric_records,
    load_incremental_run_records,
    normalize_result_table,
    render_forecast_comparison_pack,
    render_benchmark_result_analysis_pack,
)
from benchmark.industrial.evaluation.evolution import (
    build_evolution_dynamics_frame,
    load_composition_history,
    render_evolution_analysis_pack,
    select_notable_pipelines,
)


def test_result_analysis_builds_ranks_topk_and_deltas(tmp_path: Path) -> None:
    source = pd.DataFrame(
        [
            {"dataset_name": "D1", "Industrial": 0.91, "SOTA": 0.88, "Baseline": 0.72},
            {"dataset_name": "D2", "Industrial": 0.75, "SOTA": 0.80, "Baseline": 0.70},
        ]
    )
    spec = ResultAnalysisSpec(metric_name="accuracy", metric_direction="higher", source_label="unit")

    long = normalize_result_table(source, spec=spec)
    best = build_best_per_dataset_frame(long)
    ranks = build_mean_rank_frame(long)
    topk = build_topk_summary_frame(long, top_k=(1, 2))
    deltas = build_dataset_delta_frame(long, target_model="Industrial")
    manifest = render_benchmark_result_analysis_pack(
        long,
        tmp_path / "pack",
        spec=spec,
        target_model="Industrial",
    )

    assert set(long["model_name"]) == {"Industrial", "SOTA", "Baseline"}
    assert best.loc[best["dataset_name"] == "D1", "best_model"].item() == "Industrial"
    assert ranks.loc[ranks["model_name"] == "Industrial", "mean_rank"].item() == 1.5
    assert topk.loc[topk["model_name"] == "Industrial", "top_1"].item() == 1
    assert deltas.loc[deltas["dataset_name"] == "D2", "improvement"].item() < 0
    assert (tmp_path / "pack" / "summary.md").is_file()
    assert any(record.kind == "plot" for record in manifest)


def test_incremental_ingestion_coverage_and_model_diagnostics(tmp_path: Path) -> None:
    records = tmp_path / "run" / "records"
    records.mkdir(parents=True)
    _append_jsonl(
        records / "metrics.jsonl",
        {
            "run_id": "run_1",
            "benchmark": "ucr",
            "dataset_name": "D1",
            "subset": "default",
            "series_id": "D1",
            "model_name": "KernelModel",
            "metric_name": "accuracy",
            "metric_value": 0.9,
            "status": "success",
            "horizon_index": None,
        },
    )
    _append_jsonl(
        records / "runs.jsonl",
        {
            "run_id": "run_1",
            "benchmark": "ucr",
            "dataset_name": "D1",
            "subset": "default",
            "series_id": "D1",
            "model_name": "KernelModel",
            "status": "success",
            "tags": ["industrial"],
            "message": "",
            "metrics_summary": {"accuracy": 0.9},
            "metadata": {
                "kernel_learning_summary": {
                    "selected_generators": ["quantile_extractor_torch"],
                    "important_generators": ["quantile_extractor_torch"],
                    "n_kernels": 1,
                }
            },
        },
    )

    metrics = load_incremental_metric_records(
        tmp_path,
        metric_name="accuracy",
        source_label="unit",
        task_type="ts_classification",
        metric_direction="higher",
    )
    runs = load_incremental_run_records(tmp_path)
    coverage = build_coverage_frame(metrics, expected_dataset_count=2, source_label="unit")
    diagnostics = build_model_diagnostics_frame(runs)
    generators = build_generator_usage_frame(diagnostics)

    assert metrics.iloc[0]["metric_value"] == 0.9
    assert coverage.iloc[0]["status"] == "partial"
    assert diagnostics.iloc[0]["selected_generator_count"] == 1
    assert generators.iloc[0]["generator_name"] == "quantile_extractor_torch"


def test_evolution_parser_and_report_pack_use_composition_fixture(tmp_path: Path) -> None:
    root = tmp_path / "composition_results"
    _write_pipeline(root / "DatasetA" / "0" / "pipeline_a.json", fitness=0.42, nodes=("lagged", "ridge"))
    _write_pipeline(root / "DatasetA" / "1" / "pipeline_b.json", fitness=0.31, nodes=("quantile", "ridge"))
    _write_pipeline(root / "DatasetB" / "0" / "pipeline_c.json", fitness=0.55, nodes=("wavelet", "xgb"))

    history = load_composition_history(root)
    dynamics = build_evolution_dynamics_frame(history)
    operations = build_operation_frequency_frame(history)
    notable = select_notable_pipelines(history, per_dataset=1)
    manifest = render_evolution_analysis_pack(root, tmp_path / "evolution_pack")

    assert len(history) == 3
    assert set(history["dataset_name"]) == {"DatasetA", "DatasetB"}
    assert dynamics.loc[dynamics["dataset_name"] == "DatasetA", "best_fitness"].min() == 0.31
    assert operations.loc[operations["operation_name"] == "ridge", "pipeline_count"].item() == 2
    assert notable.groupby("dataset_name").size().max() == 1
    assert (tmp_path / "evolution_pack" / "summary.md").is_file()
    assert any(record.kind == "plot" for record in manifest)


def test_forecast_comparison_pack_renders_multi_model_forecasts(tmp_path: Path) -> None:
    history = (1.0, 2.0, 3.0, 4.0)
    actual = (5.0, 6.0)
    forecasts = {
        "LaggedRidgeForecaster": (4.8, 5.9),
        "KernelEnsembleForecaster_identity_shapelet": (5.1, 6.1),
    }

    comparison = build_forecast_comparison_frame(history=history, actual=actual, forecasts=forecasts)
    metrics = build_forecast_metric_frame(actual=actual, forecasts=forecasts)
    manifest = render_forecast_comparison_pack(
        history=history,
        actual=actual,
        forecasts=forecasts,
        output_dir=tmp_path / "forecast_pack",
        source_metadata={"source_kind": "unit_fixture"},
    )

    assert set(comparison["model_name"]) == {"actual_history", "actual", *forecasts}
    assert metrics.iloc[0]["mae"] <= metrics.iloc[-1]["mae"]
    assert (tmp_path / "forecast_pack" / "plots" / "multi_model_forecast.png").is_file()
    assert (tmp_path / "forecast_pack" / "source_metadata.json").is_file()
    assert any(record.kind == "plot" for record in manifest)


def test_forecast_comparison_from_progress_items_requires_real_predictions(tmp_path: Path) -> None:
    item_dir = tmp_path / "progress" / "items"
    item_dir.mkdir(parents=True)
    item_path = item_dir / "series__model.json"
    item_path.write_text(
        json.dumps(
            {
                "series_record": {
                    "dataset_name": "m4_daily",
                    "series_id": "S1",
                    "subset": "Daily",
                    "train_values": [1.0, 2.0],
                    "test_values": [3.0, 4.0],
                },
                "run_record": {"model_name": "ModelA"},
                "prediction_records": [
                    {"horizon_index": 0, "y_pred": 2.9, "status": "success"},
                    {"horizon_index": 1, "y_pred": 4.2, "status": "success"},
                ],
            }
        ),
        encoding="utf-8",
    )

    history, actual, forecasts, metadata = build_forecast_comparison_from_progress_items(tmp_path, series_id="S1")
    render_forecast_comparison_pack(
        history=history,
        actual=actual,
        forecasts=forecasts,
        output_dir=tmp_path / "pack",
        source_metadata=metadata,
    )
    digest = file_md5(tmp_path / "pack" / "plots" / "multi_model_forecast.png")

    assert history == (1.0, 2.0)
    assert actual == (3.0, 4.0)
    assert forecasts["ModelA"] == (2.9, 4.2)
    assert len(digest) == 32


def _write_pipeline(path: Path, *, fitness: float, nodes: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "uid": path.stem,
        "fitness": {"_values": [fitness]},
        "metadata": {"computation_time_in_seconds": 1.5},
        "graph": {
            "operator": {
                "_nodes": [
                    {
                        "uid": f"node_{index}",
                        "_nodes_from": [f"node_{index - 1}"] if index else [],
                        "content": {"name": name},
                    }
                    for index, name in enumerate(nodes)
                ]
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")

from __future__ import annotations

import json
from pathlib import Path

from benchmark.v2.core import (
    ArtifactSpec,
    BenchmarkAggregateReport,
    BenchmarkSuiteConfig,
    ClassificationBenchmarkResult,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
)
from fedot_ind.core.kernel_learning.experiments_api import (
    KernelLearningStage1Runner,
    KernelLearningStage2Runner,
    build_stage2_initial_population,
    importance_report_from_selection,
    load_stage1_result_from_artifacts,
)
from fedot_ind.core.kernel_learning.experiments_api import stage1 as stage1_module


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(payload, ensure_ascii=False) for payload in payloads),
        encoding="utf-8",
    )


def test_stage1_artifacts_loader_restores_classification_result_from_artifacts(tmp_path):
    run_dir = tmp_path / "stage1_root" / "kernel_learning_ucr_stage1_deadbeef"
    _write_text(
        run_dir / "aggregate" / "run_metadata.json",
        json.dumps({"run_id": run_dir.name, "task_type": "ts_classification"}),
    )
    _write_text(
        run_dir / "aggregate" / "runs.csv",
        "\n".join(
            [
                "run_id,benchmark,dataset_name,subset,model_name,status,accuracy,f1_macro",
                f"{run_dir.name},ucr,Coffee,default,KernelEnsemble,success,0.9,0.8",
            ]
        ),
    )
    _write_text(
        run_dir / "aggregate" / "metrics.csv",
        "\n".join(
            [
                "run_id,benchmark,dataset_name,subset,series_id,model_name,metric_name,metric_value,status",
                f"{run_dir.name},ucr,Coffee,default,Coffee,KernelEnsemble,f1_macro,0.8,success",
            ]
        ),
    )
    _write_text(
        run_dir / "aggregate" / "leaderboard.csv",
        "rank,model_name,dataset_name,f1_macro\n1,KernelEnsemble,Coffee,0.8",
    )
    _write_jsonl(
        run_dir / "records" / "kernel_selection.jsonl",
        [
            {
                "dataset_name": "Coffee",
                "model_name": "KernelEnsemble",
                "kernel_selection": {
                    "important_generators": ["wavelet_extractor"],
                    "important_weights": [0.8],
                },
            },
        ],
    )

    result = load_stage1_result_from_artifacts(run_dir, data_root=tmp_path / "data")

    assert result.run_id == run_dir.name
    assert result.config.artifact_spec.output_dir == str(run_dir.parent)
    assert result.config.datasets[0].dataset_name == "Coffee"
    assert result.config.models[0].adapter_name == "kernel_ensemble_classifier"
    assert result.config.models[0].display_name == "KernelEnsemble"
    assert result.run_records[0].metrics_summary == {"accuracy": 0.9, "f1_macro": 0.8}
    assert result.metric_records[0].metric_name == "f1_macro"
    assert result.aggregate_report.status_counts == {"success": 1}


def test_stage1_runner_build_config_uses_local_discovery_when_datasets_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage1_module,
        "discover_local_ucr_datasets",
        lambda data_root, allowed_names: ("Coffee", "Lightning7"),
    )

    config = KernelLearningStage1Runner(
        data_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        datasets=(),
    ).build_config()

    assert tuple(spec.dataset_name for spec in config.datasets) == ("Coffee", "Lightning7")
    assert all(spec.adapter_options["download_if_missing"] for spec in config.datasets)
    assert all(spec.adapter_options["local_data_root"] == str(tmp_path / "data") for spec in config.datasets)
    assert config.models[0].params["generator_names"] == stage1_module.DEFAULT_STAGE1_GENERATORS
    assert config.models[0].params["torch_device"] == "auto"
    assert config.artifact_spec.persist_on_run


def test_importance_report_from_selection_prefers_saved_importance_items():
    report = importance_report_from_selection(
        {
            "kernel_importance": {
                "items": [
                    {"name": "wavelet_extractor", "weight": 0.7, "original_index": 2, "rank": 1},
                    {"name": "fourier_extractor", "weight": 0.2, "original_index": 0, "rank": 2},
                ],
                "diagnostics": {"source": "stage1"},
            }
        }
    )

    assert report.selected_generators == ("wavelet_extractor", "fourier_extractor")
    assert report.selected_weights == (0.7, 0.2)
    assert tuple(item.original_index for item in report.items) == (2, 0)
    assert report.items[0].selected_by == "saved_importance"
    assert report.diagnostics == {"source": "stage1"}


def test_importance_report_from_selection_falls_back_to_selected_generators():
    report = importance_report_from_selection(
        {
            "selected_generators": ["quantile_extractor_torch", "recurrence_extractor"],
            "selected_weights": [0.4, 0.6],
        }
    )

    assert report.selected_generators == ("quantile_extractor_torch", "recurrence_extractor")
    assert report.selected_weights == (0.4, 0.6)
    assert tuple(item.selected_by for item in report.items) == ("saved_selection", "saved_selection")
    assert report.diagnostics["source"] == "kernel_selection_artifact"


def test_build_stage2_initial_population_returns_lazy_builders_by_default():
    _, specs, initial_population = build_stage2_initial_population(
        {
            "selected_generators": ["wavelet_basis"],
            "selected_weights": [1.0],
        }
    )

    assert specs[0].operation_names == ("wavelet_basis", "quantile_extractor_torch", "rf")
    assert initial_population[0].__class__.__name__ == "PipelineBuilder"
    assert callable(getattr(initial_population[0], "build"))


def test_stage2_runner_load_dataset_uses_benchmark_adapter_resolution(tmp_path):
    runner = KernelLearningStage2Runner(output_dir=tmp_path / "stage2")
    dataset = runner._load_dataset(
        DatasetSpec(
            benchmark="in_memory_tsc",
            dataset_name="toy",
            adapter_options={
                "record": {
                    "train_features": [[0.0], [1.0]],
                    "train_target": ["a", "b"],
                    "test_features": [[0.2], [1.2]],
                    "test_target": ["a", "b"],
                }
            },
        )
    )

    assert dataset["train_x"].shape == (2, 1)
    assert dataset["test_x"].shape == (2, 1)
    assert dataset["train_y"].tolist() == ["a", "b"]


def test_stage2_runner_run_uses_existing_stage1_artifacts_without_rerunning_stage1(tmp_path, monkeypatch):
    stage1_root = tmp_path / "stage1"
    run_id = "kernel_learning_ucr_stage1_existing"
    _write_jsonl(
        stage1_root / run_id / "records" / "kernel_selection.jsonl",
        [
            {
                "dataset_name": "Coffee",
                "model_name": "KernelEnsemble",
                "kernel_selection": {"selected_generators": ["wavelet_extractor"], "selected_weights": [1.0]},
            },
        ],
    )
    dataset_specs = (
        DatasetSpec(benchmark="ucr", dataset_name="Coffee"),
        DatasetSpec(benchmark="ucr", dataset_name="Lightning7"),
    )
    stage1_result = ClassificationBenchmarkResult(
        run_id=run_id,
        config=BenchmarkSuiteConfig(
            task_type=TaskType.TS_CLASSIFICATION,
            datasets=dataset_specs,
            models=(ModelSpec(adapter_name="kernel_ensemble_classifier", display_name="KernelEnsemble"),),
            artifact_spec=ArtifactSpec(output_dir=str(stage1_root), persist_on_run=True),
            run_spec=RunSpec(run_name="kernel_learning_ucr_stage1", primary_metric="f1_macro"),
            metrics=("f1_macro",),
        ),
        dataset_records=(),
        run_records=(),
        prediction_records=(),
        metric_records=(),
        aggregate_report=BenchmarkAggregateReport(
            run_id=run_id,
            task_type=TaskType.TS_CLASSIFICATION,
            primary_metric="f1_macro",
            leaderboard_rows=(),
            status_counts={},
        ),
    )
    runner = KernelLearningStage2Runner(output_dir=tmp_path / "stage2")
    calls = []

    def fake_iter_over_dataset(dataset_spec, kernel_record):
        calls.append((dataset_spec.dataset_name, kernel_record["dataset_name"]))
        return {"dataset_name": dataset_spec.dataset_name, "status": "success"}

    monkeypatch.setattr(runner, "iter_over_dataset", fake_iter_over_dataset)

    result = runner.run(stage1_result)

    assert calls == [("Coffee", "Coffee")]
    assert result == ({"dataset_name": "Coffee", "status": "success"},)
    assert (tmp_path / "stage2" / "stage2_summary.json").exists()

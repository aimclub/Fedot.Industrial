from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark.industrial import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
    discover_local_ucr_datasets,
    run_tsc_benchmark_suite,
)
from benchmark.industrial.experiments.artifacts import sanitize_artifact_payload
from benchmark.industrial.models.kernel_artifacts import export_kernel_learning_artifacts
from fedot_ind.core.kernel_learning.contracts import KernelBundle, KernelSelectionReport
from fedot_ind.core.kernel_learning.selection import KernelImportanceConfig, select_significant_generators


def _in_memory_dataset(name: str) -> DatasetSpec:
    return DatasetSpec(
        benchmark="in_memory_tsc",
        dataset_name=name,
        adapter_options={
            "record": {
                "train_features": np.array([[0.0], [1.0], [1.1]]),
                "train_target": np.array(["a", "b", "b"]),
                "test_features": np.array([[0.2], [1.2]]),
                "test_target": np.array(["a", "b"]),
            },
        },
    )


def test_tsc_suite_writes_incremental_artifacts_for_each_run(tmp_path: Path):
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(_in_memory_dataset("toy_a"), _in_memory_dataset("toy_b")),
        models=(ModelSpec(adapter_name="majority_class", display_name="MajorityClass"),),
        metrics=("accuracy", "balanced_accuracy", "f1_macro"),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name="incremental_tsc", primary_metric="accuracy", show_progress=False),
    )

    result = run_tsc_benchmark_suite(config)
    root = tmp_path / result.run_id

    for dataset_name in ("toy_a", "toy_b"):
        run_dir = root / "runs" / dataset_name / "MajorityClass"
        assert (run_dir / "run.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.csv").exists()

    runs_jsonl = root / "records" / "runs.jsonl"
    assert runs_jsonl.exists()
    assert len(runs_jsonl.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert (root / "records" / "metrics.jsonl").exists()
    assert (root / "records" / "predictions.jsonl").exists()
    assert (root / "aggregate" / "runs.csv").exists()


def test_kernel_artifact_export_summarizes_selection_and_kernel_shapes():
    report = KernelSelectionReport(
        generator_names=("informative", "noise"),
        weights=(0.9, 0.1),
        selected_generators=("informative", "noise"),
        selected_weights=(0.9, 0.1),
        scores={"informative": 0.9, "noise": 0.1},
        alignments={"informative": 0.9, "noise": 0.1},
        complexities={"informative": 1.0, "noise": 1.0},
        redundancies={"informative": 0.0, "noise": 0.2},
        task_type="classification",
    )
    model = SimpleNamespace(
        selection_report_=report,
        kernel_importance_=select_significant_generators(report, KernelImportanceConfig(weight_threshold=0.5)),
        selected_generators_=("informative", "noise"),
        selected_weights_=(0.9, 0.1),
        important_generators_=("informative",),
        important_weights_=(0.9,),
        kernel_bundles_=[
            KernelBundle(
                name="informative",
                train_kernel=np.eye(3),
                diagnostics={"min_eigenvalue": 1.0, "condition_number": 1.0},
                complexity={"trace": 3.0},
                psd_correction="clip",
            ),
        ],
        last_test_kernel_bundles_=[
            KernelBundle(name="informative", train_kernel=np.eye(3), test_kernel=np.ones((2, 3))),
        ],
    )

    artifacts = export_kernel_learning_artifacts(model)

    assert artifacts["kernel_selection"]["important_generators"] == ["informative"]
    assert artifacts["kernel_diagnostics"]["kernels"][0]["train_kernel_shape"] == [3, 3]
    assert artifacts["kernel_diagnostics"]["kernels"][0]["test_kernel_shape"] == [2, 3]


def test_sanitize_artifact_payload_replaces_non_finite_numbers():
    payload = sanitize_artifact_payload({"ok": np.float64(1.0), "bad": np.inf, "nested": [np.nan]})

    assert payload == {"ok": 1.0, "bad": None, "nested": [None]}


def test_discover_local_ucr_datasets_filters_allowed_supported_splits(tmp_path: Path):
    for name in ("Coffee", "Lightning7"):
        dataset_dir = tmp_path / name
        dataset_dir.mkdir()
        (dataset_dir / f"{name}_TRAIN.tsv").write_text("0\t1\t2\n", encoding="utf-8")
        (dataset_dir / f"{name}_TEST.tsv").write_text("0\t1\t2\n", encoding="utf-8")
    unsupported = tmp_path / "Broken"
    unsupported.mkdir()
    (unsupported / "Broken_TRAIN.tsv").write_text("0\t1\t2\n", encoding="utf-8")

    discovered = discover_local_ucr_datasets(tmp_path, allowed_names=("Lightning7", "Coffee", "Broken"))

    assert discovered == ("Coffee", "Lightning7")

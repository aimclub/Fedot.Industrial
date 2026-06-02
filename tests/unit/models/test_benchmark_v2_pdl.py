from pathlib import Path

import numpy as np
import pytest

from benchmark.v2 import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    RunStatus,
    TaskType,
    run_tsc_benchmark_suite,
)

pytest.importorskip("fedot")


def _small_multiclass_record() -> dict:
    train_features = []
    train_target = []
    test_features = []
    test_target = []
    for class_id in range(5):
        center = float(class_id * 3)
        train_features.extend(
            [
                np.linspace(center, center + 1.0, 12),
                np.linspace(center + 0.2, center + 1.2, 12),
            ]
        )
        train_target.extend([f"class_{class_id}", f"class_{class_id}"])
        test_features.append(np.linspace(center + 0.1, center + 1.1, 12))
        test_target.append(f"class_{class_id}")
    return {
        "train_features": np.asarray(train_features, dtype=float),
        "train_target": np.asarray(train_target, dtype=object),
        "test_features": np.asarray(test_features, dtype=float),
        "test_target": np.asarray(test_target, dtype=object),
    }


def test_pdl_classification_adapters_run_and_export_diagnostics(tmp_path: Path):
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(
            DatasetSpec(
                benchmark="in_memory_tsc",
                dataset_name="toy_pdl_multiclass",
                adapter_options={"record": _small_multiclass_record()},
            ),
        ),
        models=(
            ModelSpec(
                adapter_name="quantile_rf_classifier",
                display_name="QuantileRF",
                params={"quantile_params": {"window_size": 4}, "n_estimators": 5, "random_state": 42},
            ),
            ModelSpec(
                adapter_name="pdl_classifier",
                display_name="PDLRaw",
                params={"model": "rf", "backend": "numpy", "max_pairs": 200, "n_estimators": 5},
            ),
            ModelSpec(
                adapter_name="pdl_quantile_classifier",
                display_name="PDLQuantile",
                params={
                    "model": "rf",
                    "backend": "numpy",
                    "max_pairs": 200,
                    "n_estimators": 5,
                    "quantile_params": {"window_size": 4},
                },
            ),
        ),
        metrics=("accuracy", "balanced_accuracy", "f1_macro"),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name="toy_pdl", primary_metric="f1_macro", show_progress=False),
    )

    result = run_tsc_benchmark_suite(config)

    assert len(result.run_records) == 3
    assert all(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert all("model_diagnostics" in record.metadata["artifact_paths"] for record in result.run_records)
    diagnostics_records = tmp_path / result.run_id / "records" / "model_diagnostics.jsonl"
    assert diagnostics_records.exists()
    assert "pair_feature_dim" in diagnostics_records.read_text(encoding="utf-8")


def test_pdl_small_multiclass_script_declares_expected_experiment_contract():
    script = Path("benchmark/run_pdl_small_multiclass_ucr.py").read_text(encoding="utf-8")

    assert "UNI_CLF_BENCH" in script
    assert "MULTI_CLF_BENCH" in script
    assert "ModelSpec" in script
    assert "quantile_rf_classifier" in script
    assert "pdl_classifier" in script
    assert "pdl_quantile_classifier" in script
    assert "persist_on_run=True" in script

from benchmark.v2.classification import (
    BenchmarkClassificationError,
    LocalClassificationAdapter,
)
from benchmark.v2.api import build_legacy_tsc_suite_config, build_legacy_tser_suite_config
from benchmark.v2 import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    RunStatus,
    TaskType,
    run_tsc_benchmark_suite,
    run_tser_benchmark_suite,
)
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fedot")


def test_tsc_suite_runs_kernel_ensemble_adapter(tmp_path: Path):
    record = {
        "train_features": np.array([[0.0], [0.2], [1.0], [1.2]]),
        "train_target": np.array(["a", "a", "b", "b"]),
        "test_features": np.array([[0.1], [1.1]]),
        "test_target": np.array(["a", "b"]),
    }
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(
            DatasetSpec(
                benchmark="in_memory_tsc",
                dataset_name="toy_kernel_tsc",
                adapter_options={"record": record},
            ),
        ),
        models=(
            ModelSpec(
                adapter_name="kernel_ensemble_classifier",
                display_name="KernelEnsembleClassifier",
                params={"generator_names": ("statistical_summary",), "kernel": "linear", "C": 10.0},
            ),
        ),
        metrics=("accuracy", "balanced_accuracy", "f1_macro"),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name="toy_kernel_tsc", primary_metric="accuracy"),
    )

    result = run_tsc_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert result.run_records[0].model_name == "KernelEnsembleClassifier"


def test_tser_suite_runs_kernel_ensemble_adapter(tmp_path: Path):
    record = {
        "train_features": np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        "train_target": np.array([1.0, 3.0, 5.0, 7.0, 9.0]),
        "test_features": np.array([[5.0], [6.0]]),
        "test_target": np.array([11.0, 13.0]),
    }
    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_REGRESSION,
        datasets=(
            DatasetSpec(
                benchmark="in_memory_tser",
                dataset_name="toy_kernel_tser",
                adapter_options={"record": record},
            ),
        ),
        models=(
            ModelSpec(
                adapter_name="kernel_ensemble_regressor",
                display_name="KernelEnsembleRegressor",
                params={"generator_names": ("statistical_summary",), "kernel": "linear", "alpha": 1e-6},
            ),
        ),
        metrics=("rmse", "mae", "r2"),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name="toy_kernel_tser", primary_metric="rmse"),
    )

    result = run_tser_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert result.run_records[0].model_name == "KernelEnsembleRegressor"


def test_legacy_builders_default_to_kernel_model_specs():
    tsc_config = build_legacy_tsc_suite_config({"custom_datasets": ("Lightning7",)})
    tser_config = build_legacy_tser_suite_config({"custom_datasets": ("NaturalGasPricesSentiment",)})

    assert tsc_config.models[0].adapter_name == "kernel_ensemble_classifier"
    assert tsc_config.datasets[0].dataset_name == "Lightning7"
    assert tser_config.models[0].adapter_name == "kernel_ensemble_regressor"
    assert tser_config.datasets[0].dataset_name == "NaturalGasPricesSentiment"


def test_ucr_adapter_uses_dataloader_fallback_with_local_root(monkeypatch, tmp_path: Path):
    captured = {}

    class FakeDataLoader:
        def __init__(self, dataset_name, folder=None):
            captured["dataset_name"] = dataset_name
            captured["folder"] = folder

        def load_data(self):
            train = (np.array([[0.0], [1.0]]), np.array(["a", "b"]))
            test = (np.array([[0.2], [1.2]]), np.array(["a", "b"]))
            return train, test

    fake_loader_module = types.ModuleType("fedot_ind.tools.loader")
    fake_loader_module.DataLoader = FakeDataLoader
    monkeypatch.setitem(sys.modules, "fedot_ind.tools.loader", fake_loader_module)
    spec = DatasetSpec(
        benchmark="ucr",
        dataset_name="MissingUCR",
        adapter_options={"local_data_root": str(tmp_path), "download_if_missing": True},
    )

    records = LocalClassificationAdapter().load_dataset(spec)

    assert captured == {"dataset_name": "MissingUCR", "folder": str(tmp_path)}
    assert records[0].metadata["split_provenance"] == "fedot_ind.tools.loader"


def test_ucr_adapter_can_disable_download_fallback(tmp_path: Path):
    spec = DatasetSpec(
        benchmark="ucr",
        dataset_name="MissingUCR",
        adapter_options={"local_data_root": str(tmp_path), "download_if_missing": False},
    )

    with pytest.raises(BenchmarkClassificationError):
        LocalClassificationAdapter().load_dataset(spec)

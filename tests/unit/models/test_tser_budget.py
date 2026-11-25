from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.industrial import regression
from benchmark.industrial.core import (
    ArtifactSpec, BenchmarkRunRecord, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, RunStatus, TaskType,
)
from benchmark.industrial.models.tser_budget import PANEL_BUDGET_POLICY, PanelShape, panel_budget_rejection


@pytest.mark.parametrize("adapter", [
    "drcif_regressor", "dtw_1nn_regressor", "freshprince_regressor", "rocket_ridge_regressor",
])
def test_panel_guard_admits_the_audit_fixture_under_default_parameters(adapter):
    assert panel_budget_rejection(adapter, PanelShape(100, 20, 3, 100), {}) == ""


@pytest.mark.parametrize("adapter,shape,params,reason", [
    ("dtw_1nn_regressor", PanelShape(100, 100, 3, 100), {}, "DTW alignment work"),
    ("dtw_1nn_regressor", PanelShape(2, 1, 1, 6000), {}, "DTW matrix cells"),
    ("rocket_ridge_regressor", PanelShape(100, 20, 20, 100), {}, "ROCKET convolution work"),
    ("rocket_ridge_regressor", PanelShape(1700, 10, 1, 24), {}, "ROCKET transformed cells"),
    ("rocket_ridge_regressor", PanelShape(650, 10, 1, 10), {}, "RidgeCV decomposition work"),
    ("freshprince_regressor", PanelShape(100, 20, 3, 100), {"n_estimators": 10_000}, "FreshPRINCE feature"),
    ("drcif_regressor", PanelShape(100, 20, 3, 100), {"n_estimators": 10_000}, "DrCIF interval feature work"),
    ("drcif_regressor", PanelShape(100, 20, 3, 100),
     {"n_estimators": 2, "time_limit_in_minutes": 10, "contract_max_n_estimators": 10_000},
     "DrCIF interval feature work"),
    ("drcif_regressor", PanelShape(2000, 2, 30, 1000), {}, "panel input cells"),
])
def test_panel_guard_rejects_expensive_work_without_fitting(adapter, shape, params, reason):
    assert reason in panel_budget_rejection(adapter, shape, params)


def test_unrecognised_distance_and_flat_adapters_keep_legacy_guard():
    shape = PanelShape(100, 20, 3, 100)
    assert panel_budget_rejection("dtw_1nn_regressor", shape, {"distance": "euclidean"}) is None
    assert panel_budget_rejection("flat_forest_regressor", shape, {}) is None


def _run_record(model_name, status=RunStatus.NOT_AVAILABLE, **metadata):
    return BenchmarkRunRecord(
        run_id="prior", benchmark="in_memory_tser", dataset_name="Toy", subset="default", series_id="Toy",
        model_name=model_name, status=status, metadata=metadata,
    )


def test_resume_only_refreshes_skips_from_the_old_panel_guard():
    runs = {
        "old panel": _run_record("old panel", failure_stage="runtime_guard"),
        "new panel": _run_record("new panel", failure_stage="runtime_guard", runtime_guard_policy=PANEL_BUDGET_POLICY),
        "missing dependency": _run_record("missing dependency"),
        "successful panel": _run_record("successful panel", RunStatus.SUCCESS),
        "flat": _run_record("flat", failure_stage="runtime_guard"),
    }
    records = {("Toy", "default", name): record for name, record in runs.items()}
    models = tuple(ModelSpec("flat_forest_regressor" if name == "flat" else "drcif_regressor", name) for name in runs)
    assert set(regression._reusable_regression_runs(records, models)
               ) == set(records) - {("Toy", "default", "old panel")}


def test_suite_fits_the_previously_skipped_panel_but_keeps_guarding_heavy_and_flat_models(tmp_path, monkeypatch):
    models = (
        ModelSpec("drcif_regressor", "small-panel"),
        ModelSpec("drcif_regressor", "large-panel", params={"n_estimators": 10_000}),
        ModelSpec("flat_forest_regressor", "flat"),
    )
    record = {
        "train_features": np.zeros((100, 3, 100)), "train_target": np.zeros(100),
        "test_features": np.zeros((20, 3, 100)), "test_target": np.zeros(20),
    }
    config = BenchmarkSuiteConfig(
        TaskType.TS_REGRESSION,
        (DatasetSpec("in_memory_tser", "Toy", adapter_options={"record": record}),),
        models, ArtifactSpec(str(tmp_path / "out"), persist_on_run=False),
        RunSpec(primary_metric="rmse", show_progress=False, resume_enabled=True), metrics=("rmse",),
    )
    old_skip = _run_record("small-panel", failure_stage="runtime_guard")
    monkeypatch.setattr(regression, "load_incremental_run_records",
                        lambda *_: {("Toy", "default", "small-panel"): old_skip})
    fitted = []

    def model(spec):
        return SimpleNamespace(
            name=spec.display_name, tags=(), uses_panel=spec.adapter_name == "drcif_regressor",
            availability=lambda: (RunStatus.SUCCESS, "ready"),
            fit=lambda features, target: fitted.append((spec.display_name, features.shape, target.shape)),
            predict=lambda features: np.zeros(len(features)),
        )

    monkeypatch.setattr(regression, "build_regression_model", model)
    small, large, flat = regression.run_tser_suite(config).run_records
    assert fitted == [("small-panel", (100, 3, 100), (100,))]
    assert small.status is RunStatus.SUCCESS and small.metrics_summary["rmse"] == 0.
    assert large.status is flat.status is RunStatus.NOT_AVAILABLE
    assert "DrCIF interval feature work" in large.message
    assert large.metadata["runtime_guard_policy"] == PANEL_BUDGET_POLICY
    assert "wide local TSER fit" in flat.message

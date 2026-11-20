import numpy as np
import pytest

from benchmark.industrial.core import ModelSpec, RunStatus
from benchmark.industrial.datasets.local_io import load_local_supervised_split
from benchmark.industrial.errors import BenchmarkRegressionError
from benchmark.industrial.models.regression import build_regression_model
from benchmark.industrial.models.tser import restore_tser_panel
from benchmark.industrial.regression import _encode_dataset_record, compute_regression_metric


def test_panel_metadata_and_flat_euclidean_control():
    panel = np.arange(72.).reshape(6, 2, 6)
    target = np.arange(6.)
    data = _encode_dataset_record("ts", "demo", "small", panel, target, panel, target)
    assert data.metadata["dimensions"] == 2
    np.testing.assert_array_equal(restore_tser_panel(data.train_features, 2), panel)
    model = build_regression_model(ModelSpec(adapter_name="ed_1nn_regressor", display_name="ED"))
    model.fit(np.asarray(data.train_features), target)
    assert compute_regression_metric("rmse", target, model.predict(data.test_features)) == 0.
    with pytest.raises(BenchmarkRegressionError, match="channel"):
        restore_tser_panel(data.train_features, 5)


def test_ts_parser_detects_ragged_channels_in_flat_rows(tmp_path):
    train = tmp_path / "Toy_TRAIN.ts"
    control = tmp_path / "Toy_TEST.ts"
    header = "@problemName Toy\n@timestamps false\n@targetlabel true\n@data\n"
    train.write_text(header + "1,2:3,4:0.1\n", encoding="utf-8")
    control.write_text(header + "1:2,3,4:0.2\n", encoding="utf-8")
    data = load_local_supervised_split("Toy", train_path=train, test_path=control)
    assert data.train_features.shape == data.test_features.shape == (1, 4)
    assert data.metadata["dimensions"] == 2
    assert not data.metadata["uniform_channel_layout"]


@pytest.mark.parametrize("left,right,aligned", [
    ("(0,1),(2,2),(4,3)", "(0,4),(2,5),(4,6)", True),
    ("(0,1),(2,2),(4,3)", "(1,4),(3,5),(5,6)", False),
    ("(0,1),(2,2),(4,3)", "(0,4),(4,6),(2,5)", False),
    ("(0,1),(0,2),(4,3)", "(0,4),(0,5),(4,6)", False),
    ("(0,1),(2,2),(5,3)", "(0,4),(2,5),(5,6)", False),
])
def test_panel_examples_check_timestamps_not_only_channel_sizes(tmp_path, left, right, aligned):
    from examples.release_1_0.tsf_panel_validation import load_train

    train = tmp_path / "Toy_TRAIN.ts"
    control = tmp_path / "Toy_TEST.ts"
    header = "@problemName Toy\n@timestamps true\n@targetlabel true\n@data\n"
    for file in (train, control):
        file.write_text(f"{header}{left}:{right}:0.1\n", encoding="utf-8")

    data = load_local_supervised_split("Toy", train_path=train, test_path=control)
    assert data.train_features.shape == (1, 6)
    assert data.metadata["uniform_channel_layout"]
    assert data.metadata["aligned_channel_grid"] is aligned
    if aligned:
        panel, target = load_train(train)
        np.testing.assert_array_equal(panel, [[[1, 2, 3], [4, 5, 6]]])
        np.testing.assert_array_equal(target, [.1])
    else:
        with pytest.raises(ValueError, match="aligned"):
            load_train(train)


@pytest.mark.parametrize("interval,aligned", [(2, True), (3, False)])
def test_tser_requires_the_same_sampling_cadence_across_splits(tmp_path, interval, aligned):
    paths = [tmp_path / "Toy_TRAIN.ts", tmp_path / "Toy_TEST.ts"]
    header = "@problemName Toy\n@timestamps true\n@targetlabel true\n@data\n"
    for path, start, step in zip(paths, (0, 20), (2, interval)):
        # Different objects may be recorded at different absolute times.
        path.write_text(header + f"({start},1),({start + step},2):0.1\n", encoding="utf-8")
    data = load_local_supervised_split("Toy", train_path=paths[0], test_path=paths[1])
    record = _encode_dataset_record("ts", "Toy", "small", data.train_features, data.train_target,
                                    data.test_features, data.test_target, data.metadata)
    assert record.metadata["aligned_channel_grid"] is aligned
    if aligned:
        panel = restore_tser_panel(record.test_features, 1, metadata=record.metadata)
        np.testing.assert_array_equal(panel, [[[1, 2]]])
    else:
        with pytest.raises(BenchmarkRegressionError, match="aligned"):
            restore_tser_panel(record.test_features, 1, metadata=record.metadata)


def test_tser_suite_does_not_fit_panel_model_on_misaligned_grid(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmark.industrial import regression
    from benchmark.industrial.core import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, RunSpec, TaskType

    paths = [tmp_path / "Toy_TRAIN.ts", tmp_path / "Toy_TEST.ts"]
    header = "@problemName Toy\n@timestamps true\n@targetlabel true\n@data\n"
    for path in paths:
        path.write_text(header + "(0,1),(2,2):(1,3),(3,4):0.1\n", encoding="utf-8")

    def model(spec):
        if spec.adapter_name == "panel_probe":
            return SimpleNamespace(name=spec.display_name, tags=(), uses_panel=True,
                                   availability=lambda: (RunStatus.SUCCESS, "ready"),
                                   fit=lambda *_: pytest.fail("Misaligned data reached the panel estimator"))
        return build_regression_model(spec)

    monkeypatch.setattr(regression, "build_regression_model", model)
    config = BenchmarkSuiteConfig(
        TaskType.TS_REGRESSION,
        (DatasetSpec("local_tser", "Toy", adapter_options={"train_path": paths[0], "test_path": paths[1]}),),
        (ModelSpec("panel_probe", "panel"), ModelSpec("ed_1nn_regressor", "flat")),
        ArtifactSpec(str(tmp_path / "out"), persist_on_run=False),
        RunSpec(primary_metric="rmse", show_progress=False),
        metrics=("rmse",),
    )
    panel, flat = regression.run_tser_suite(config).run_records
    assert panel.status is RunStatus.FAILED and "aligned" in panel.message
    assert flat.status is RunStatus.SUCCESS and flat.metrics_summary["rmse"] == 0.


@pytest.mark.parametrize("adapter,params", [
    ("freshprince_regressor", {"default_fc_parameters": "minimal", "n_estimators": 2}),
    ("drcif_regressor", {"n_estimators": 2, "n_intervals": 1, "att_subsample_size": 3}),
    ("dtw_1nn_regressor", {}),
    ("rocket_ridge_regressor", {"n_kernels": 20}),
])
def test_optional_tser_estimator_shape_or_explicit_dependency_error(adapter, params):
    model = build_regression_model(ModelSpec(adapter_name=adapter, display_name=adapter, params=params))
    status, message = model.availability()
    if status is RunStatus.NOT_AVAILABLE:
        assert "fedot-ind[tser]" in message
        pytest.skip(message)
    x = np.random.default_rng(0).normal(size=(10, 2, 24))
    y = x.mean(axis=(1, 2))
    model.fit(x, y)
    assert model.predict(x[:2]).shape == (2,)
    x[0, 0, 0] = np.nan
    with pytest.raises(BenchmarkRegressionError, match="finite"):
        model.predict(x)

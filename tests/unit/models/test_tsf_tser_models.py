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

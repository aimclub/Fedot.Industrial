from types import SimpleNamespace

from examples.real_world_examples.benchmark_example.rkhs_okhs.forecasting.okhs_forecasting_utils import (
    build_forecaster_params,
    extract_training_history,
)


def test_build_forecaster_params_promotes_horizon_alias():
    params = build_forecaster_params({"method": "dmd"}, horizon=12, device="cpu")

    assert params["forecast_horizon"] == 12
    assert "horizon" not in params
    assert params["method"] == "dmd"
    assert params["device"] == "cpu"


def test_build_forecaster_params_preserves_explicit_forecast_horizon():
    params = build_forecaster_params({"forecast_horizon": 8}, horizon=12)

    assert params["forecast_horizon"] == 8
    assert "horizon" in params


def test_extract_training_history_returns_list_copy():
    forecaster = SimpleNamespace(dmd_model=SimpleNamespace(training_history_=[0.3, 0.2, 0.1]))

    history = extract_training_history(forecaster)

    assert history == [0.3, 0.2, 0.1]
    assert history is not forecaster.dmd_model.training_history_


def test_extract_training_history_handles_missing_dmd_model():
    forecaster = SimpleNamespace()

    history = extract_training_history(forecaster)

    assert history == []

import numpy as np

from fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head import (
    DeepARForecastHeadImplementation,
    NEURAL_FORECASTING_MODEL_REGISTRY,
    NBeatsForecastHeadImplementation,
    NeuralForecastHead,
    NeuralForecastHeadRunResult,
    NeuralForecastHeadSpec,
    PatchTSTForecastHeadImplementation,
    TCNForecastHeadImplementation,
    build_neural_forecast_head,
    build_neural_forecasting_stage_diagnostics,
    run_neural_forecast_head_on_series,
)
from fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head_bridge import (
    NeuralForecastHeadBridge,
)


def test_build_neural_forecasting_stage_diagnostics_returns_stage_vocabulary():
    diagnostics = build_neural_forecasting_stage_diagnostics(
        'patch_tst_model',
        forecast_horizon=6,
        params={'patch_len': 16, 'epochs': 20, 'batch_size': 8},
        training_history_length=96,
    )

    assert diagnostics['model_family'] == 'neural_forecaster'
    assert diagnostics['trajectory_transform']['window_size'] == 16
    assert diagnostics['forecast_head']['head_type'] == 'patch_tst_model'
    assert diagnostics['forecast_head']['forecast_horizon'] == 6


def test_neural_forecast_head_defaults_are_normalized_in_diagnostics():
    diagnostics = build_neural_forecasting_stage_diagnostics(
        'patch_tst_model',
        forecast_horizon=4,
        params={'patch_len': 12},
        training_history_length=48,
    )

    assert diagnostics['forecast_head']['epochs'] == 150
    assert diagnostics['forecast_head']['batch_size'] == 16
    assert diagnostics['forecast_head']['learning_rate'] == 1e-3
    assert diagnostics['forecast_head']['device'] == 'cuda'


def test_neural_forecast_head_wraps_native_model_contract(monkeypatch):
    class FakePrediction:
        def __init__(self, values):
            self.predict = np.asarray(values, dtype=float)

    class FakeModel:
        def __init__(self, params):
            self.params = params
            self.fit_calls = 0
            self.predict_calls = 0

        def fit(self, input_data):
            self.fit_calls += 1
            self.seen_fit_length = len(np.asarray(input_data.features).reshape(-1))
            return self

        def predict(self, input_data):
            self.predict_calls += 1
            horizon = int(input_data.task.task_params.forecast_length)
            return FakePrediction(np.linspace(1.0, float(horizon), num=horizon))

    monkeypatch.setitem(NEURAL_FORECASTING_MODEL_REGISTRY, 'patch_tst_model', FakeModel)

    head = build_neural_forecast_head(
        'patch_tst_model',
        forecast_horizon=4,
        params={'patch_len': 12, 'epochs': 2},
    )
    head.fit(np.arange(48, dtype=float))
    prediction = head.predict(np.arange(48, dtype=float))
    diagnostics = head.get_diagnostics()

    assert prediction.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert diagnostics['model_family'] == 'neural_forecaster'
    assert diagnostics['forecast_head']['head_type'] == 'patch_tst_model'
    assert diagnostics['last_prediction_diagnostics']['forecast_shape'] == (4,)


def test_neural_forecast_head_spec_canonicalizes_runtime_contract():
    spec = NeuralForecastHeadSpec(
        model_name='patch_tst_model',
        forecast_horizon=6,
        params={'patch_len': 16, 'epochs': 5},
    )

    assert spec.model_name == 'patch_tst_model'
    assert spec.forecast_horizon == 6
    assert spec.params['patch_len'] == 16
    assert spec.family == 'neural_forecaster'


def test_run_neural_forecast_head_on_series_returns_typed_result(monkeypatch):
    class FakePrediction:
        def __init__(self, values):
            self.predict = np.asarray(values, dtype=float)

    class FakeModel:
        def __init__(self, params):
            self.params = params

        def fit(self, input_data):
            self.seen_fit_length = len(np.asarray(input_data.features).reshape(-1))
            return self

        def predict(self, input_data):
            horizon = int(input_data.task.task_params.forecast_length)
            return FakePrediction(np.linspace(3.0, 3.0 + horizon - 1, num=horizon))

    monkeypatch.setitem(NEURAL_FORECASTING_MODEL_REGISTRY, 'patch_tst_model', FakeModel)

    result = run_neural_forecast_head_on_series(
        'patch_tst_model',
        time_series=np.arange(40, dtype=float),
        forecast_horizon=4,
        params={'patch_len': 12, 'epochs': 2},
    )

    assert isinstance(result, NeuralForecastHeadRunResult)
    assert result.spec.family == 'neural_forecaster'
    assert result.forecast == (3.0, 4.0, 5.0, 6.0)
    assert result.diagnostics['forecast_head']['head_type'] == 'patch_tst_model'


def test_neural_forecast_head_implementations_publish_model_specific_entrypoints():
    assert PatchTSTForecastHeadImplementation.model_name == 'patch_tst_model'
    assert TCNForecastHeadImplementation.model_name == 'tcn_model'
    assert DeepARForecastHeadImplementation.model_name == 'deepar_model'
    assert NBeatsForecastHeadImplementation.model_name == 'nbeats_model'


def test_neural_forecast_head_bridge_remains_compatibility_shell(monkeypatch):
    class FakePrediction:
        def __init__(self, values):
            self.predict = np.asarray(values, dtype=float)

    class FakeModel:
        def __init__(self, params):
            self.params = params

        def fit(self, input_data):
            self.seen_fit_length = len(np.asarray(input_data.features).reshape(-1))
            return self

        def predict(self, input_data):
            horizon = int(input_data.task.task_params.forecast_length)
            return FakePrediction(np.linspace(2.0, 2.0 + horizon - 1, num=horizon))

    monkeypatch.setitem(NEURAL_FORECASTING_MODEL_REGISTRY, 'patch_tst_model', FakeModel)

    bridge = NeuralForecastHeadBridge(
        model_name='patch_tst_model',
        forecast_horizon=3,
        params={'patch_len': 10, 'epochs': 1},
    )
    bridge.fit(np.arange(32, dtype=float))
    prediction = bridge.predict(np.arange(32, dtype=float))

    assert prediction.tolist() == [2.0, 3.0, 4.0]
    assert isinstance(bridge, NeuralForecastHead)

import numpy as np
import pytest

from fedot_ind.core.models.ts_forecasting.lagged_model.mssa_forecaster import (
    MSSAForecaster,
    MSSAForecasterImplementation,
)
from fedot_ind.core.models.ts_forecasting.regime_routing import recommend_forecasting_model
from fedot_ind.core.models.ts_forecasting.regime_utils.regime_diagnostics import analyze_regime_diagnostics


def test_mssa_forecaster_predicts_univariate_series():
    time = np.arange(120, dtype=float)
    series = np.sin(2 * np.pi * time / 12.0)
    model = MSSAForecaster(forecast_horizon=6, window_size=12, rank=2, coupled=False)

    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()

    assert forecast.shape == (6,)
    assert diagnostics['selected_rank'] >= 2
    assert diagnostics['model_family'] == 'low_rank_linear'
    assert diagnostics['trajectory_transform']['kind'] == 'page'
    assert diagnostics['decomposition']['strategy'] == 'page_svd_per_channel'
    assert diagnostics['rank_truncation']['selected_rank'] >= 2
    assert diagnostics['forecast_head']['head_type'] == 'autoregression_head'
    assert diagnostics['forecast_head']['head_policy'] == 'mlp'


def test_mssa_forecaster_supports_multivariate_series():
    time = np.arange(160, dtype=float)
    series = np.column_stack(
        [
            np.sin(2 * np.pi * time / 12.0),
            np.cos(2 * np.pi * time / 12.0),
        ]
    )
    model = MSSAForecaster(forecast_horizon=5, window_size=16, rank=3, coupled=True)

    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()

    assert forecast.shape == (5, 2)
    assert diagnostics['coupled'] is True
    assert diagnostics['decomposition']['coupled'] is True
    assert diagnostics['trajectory_transform']['channel_count'] == 2
    assert diagnostics['forecast_head']['head_policy'] == 'mlp'


def test_mssa_forecaster_supports_linear_head_fallback():
    time = np.arange(120, dtype=float)
    series = np.sin(2 * np.pi * time / 12.0)
    model = MSSAForecaster(
        forecast_horizon=4,
        window_size=12,
        rank=2,
        coupled=False,
        head_policy='linear',
    )

    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()

    assert forecast.shape == (4,)
    assert diagnostics['forecast_head']['head_policy'] == 'linear'


def test_mssa_implementation_predict_for_fit_returns_transposed_denoised_series():
    fedot = pytest.importorskip('fedot.core.data.data')
    InputData = fedot.InputData
    DataTypesEnum = pytest.importorskip('fedot.core.repository.dataset_types').DataTypesEnum
    tasks_module = pytest.importorskip('fedot.core.repository.tasks')
    Task = tasks_module.Task
    TaskTypesEnum = tasks_module.TaskTypesEnum
    TsForecastingParams = tasks_module.TsForecastingParams

    time = np.arange(100, dtype=float)
    series = np.sin(2 * np.pi * time / 10.0)
    input_data = InputData(
        idx=np.arange(len(series)),
        features=series.reshape(-1, 1),
        target=series,
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=8)),
        data_type=DataTypesEnum.ts,
    )
    model = MSSAForecasterImplementation({'window_size': 10, 'rank': 2})

    denoised = model.predict_for_fit(input_data)

    assert denoised.shape[1] == len(series)


def test_regime_diagnostics_detects_periodic_structure():
    time = np.arange(180, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0)

    diagnostics = analyze_regime_diagnostics(series).to_dict()

    assert diagnostics['dominant_period'] is not None
    assert diagnostics['spectral_concentration'] > 0.15
    assert diagnostics['switching_score'] < 0.3


def test_regime_routing_matches_periodic_diagnostics():
    time = np.arange(180, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0)

    decision = recommend_forecasting_model(analyze_regime_diagnostics(series))

    assert decision.primary_adapter in {'mssa', 'ssa_compat'}
    assert decision.rationale

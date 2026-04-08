import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.models.ts_forecasting.mssa_forecaster import (
    MSSAForecaster,
    MSSAForecasterImplementation,
)
from fedot_ind.core.models.ts_forecasting.regime_diagnostics import analyze_regime_diagnostics


def test_mssa_forecaster_predicts_univariate_series():
    time = np.arange(120, dtype=float)
    series = np.sin(2 * np.pi * time / 12.0)
    model = MSSAForecaster(forecast_horizon=6, window_size=12, rank=2, coupled=False)

    model.fit(series)
    forecast = model.predict(series)

    assert forecast.shape == (6,)
    assert model.get_diagnostics()['selected_rank'] >= 2


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

    assert forecast.shape == (5, 2)
    assert model.get_diagnostics()['coupled'] is True


def test_mssa_implementation_predict_for_fit_returns_transposed_denoised_series():
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
    assert diagnostics['regime_hint'] in {'periodic', 'locally_linear'}

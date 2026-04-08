import numpy as np
import pytest
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.models.ts_forecasting.ssa_forecaster import SSAForecasterImplementation


@pytest.fixture(scope='session')
def time_series_data():
    ts = np.sin(np.linspace(0, 4 * np.pi, 120))
    input_data = InputData(
        idx=np.arange(0, len(ts)),
        features=ts.reshape(-1, 1),
        target=ts,
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=10)),
        data_type=DataTypesEnum.ts,
    )
    return input_data


def test_ssa_predict_for_fit_returns_denoised_series(time_series_data):
    forecaster = SSAForecasterImplementation({'mode': 'one_dimensional'})

    denoised = forecaster.predict_for_fit(time_series_data)

    assert denoised is not None
    assert denoised.shape[0] == 1
    assert denoised.shape[1] == 120


def test_ssa_fit_predict_works_as_compatibility_wrapper(time_series_data):
    forecaster = SSAForecasterImplementation({'mode': 'one_dimensional'})

    forecaster.fit(time_series_data)
    prediction = forecaster.predict(time_series_data)

    assert prediction.predict is not None
    assert prediction.predict.shape[-1] == 10
    assert forecaster.compatibility_status_ == 'compatibility_wrapper'

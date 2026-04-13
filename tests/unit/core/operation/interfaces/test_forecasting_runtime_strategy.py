import numpy as np
import pytest
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.operation.interfaces.forecasting_runtime_strategy import (
    IndustrialForecastingModelRuntimeStrategy,
    IndustrialForecastingPreprocessingRuntimeStrategy,
)
from fedot_ind.core.operation.interfaces.industrial_model_strategy import IndustrialSkLearnForecastingStrategy
from fedot_ind.core.operation.interfaces.industrial_preprocessing_strategy import (
    IndustrialForecastingPreprocessingStrategy,
)


def _build_ts_input(horizon: int = 4, length: int = 64) -> InputData:
    series = np.linspace(1.0, float(length), num=length, dtype=float)
    return InputData(
        idx=np.arange(length),
        features=series,
        target=series,
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon)),
        data_type=DataTypesEnum.ts,
    )


def test_forecasting_preprocessing_runtime_strategy_hankelisation_predicts_latest_state():
    input_data = _build_ts_input()
    params = OperationParameters(window_size=12, stride=2)
    strategy = IndustrialForecastingPreprocessingRuntimeStrategy('hankelisation', params=params)

    trained_operation = strategy.fit(input_data)
    fit_output = strategy.predict_for_fit(trained_operation, input_data)
    predict_output = strategy.predict(trained_operation, input_data)

    assert isinstance(fit_output, OutputData)
    assert isinstance(predict_output, OutputData)
    assert fit_output.data_type is DataTypesEnum.table
    assert predict_output.data_type is DataTypesEnum.table
    assert fit_output.predict.ndim == 2
    assert predict_output.predict.shape[0] == 1


def test_forecasting_model_runtime_strategy_lagged_ridge_predicts_horizon_vector():
    pytest.importorskip('torch')
    input_data = _build_ts_input(horizon=5, length=96)
    params = OperationParameters(window_size=18, stride=1, alpha=1.0)
    strategy = IndustrialForecastingModelRuntimeStrategy('lagged_ridge_forecaster', params=params)

    trained_operation = strategy.fit(input_data)
    predict_output = strategy.predict(trained_operation, input_data)

    assert isinstance(predict_output, OutputData)
    assert predict_output.data_type is DataTypesEnum.table
    assert np.asarray(predict_output.predict).shape[0] == 5


def test_legacy_forecasting_strategy_wrappers_delegate_to_runtime_strategies():
    assert issubclass(IndustrialSkLearnForecastingStrategy, IndustrialForecastingModelRuntimeStrategy)
    assert issubclass(IndustrialForecastingPreprocessingStrategy, IndustrialForecastingPreprocessingRuntimeStrategy)

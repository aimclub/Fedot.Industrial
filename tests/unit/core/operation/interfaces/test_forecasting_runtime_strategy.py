import numpy as np
import pytest

fedot_data = pytest.importorskip('fedot.core.data.data')
InputData = fedot_data.InputData
OutputData = fedot_data.OutputData
OperationParameters = pytest.importorskip('fedot.core.operations.operation_parameters').OperationParameters
DataTypesEnum = pytest.importorskip('fedot.core.repository.dataset_types').DataTypesEnum
tasks_module = pytest.importorskip('fedot.core.repository.tasks')
Task = tasks_module.Task
TaskTypesEnum = tasks_module.TaskTypesEnum
TsForecastingParams = tasks_module.TsForecastingParams

from fedot_ind.core.operation.interfaces.forecasting_runtime_strategy import (
    IndustrialForecastingModelRuntimeStrategy,
    IndustrialForecastingPreprocessingRuntimeStrategy,
    LegacyForecastingModelRedirectMixin,
    LegacyForecastingPreprocessingRedirectMixin,
    should_redirect_legacy_model_strategy,
    should_redirect_legacy_preprocessing_strategy,
)
from fedot_ind.core.operation.interfaces.industrial_model_strategy import (
    FedotNNTimeSeriesStrategy as LegacyFedotNNTimeSeriesStrategy,
    IndustrialCustomRegressionStrategy,
    IndustrialSkLearnEvaluationStrategy,
    IndustrialSkLearnForecastingStrategy,
    IndustrialSkLearnRegressionStrategy,
)
from fedot_ind.core.operation.interfaces.industrial_preprocessing_strategy import (
    IndustrialCustomPreprocessingStrategy,
    IndustrialForecastingPreprocessingStrategy,
    IndustrialPreprocessingStrategy,
)
from fedot_ind.core.operation.interfaces.neural_forecasting_strategy import (
    FedotNNTimeSeriesStrategy as ExtractedFedotNNTimeSeriesStrategy,
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


def test_legacy_preprocessing_strategy_redirects_forecasting_operations_to_runtime():
    params = OperationParameters(window_size=12, stride=2)

    custom_strategy = IndustrialCustomPreprocessingStrategy('hankelisation', params=params)
    industrial_strategy = IndustrialPreprocessingStrategy('hankelisation', params=params)

    assert isinstance(custom_strategy, IndustrialForecastingPreprocessingRuntimeStrategy)
    assert isinstance(industrial_strategy, IndustrialForecastingPreprocessingRuntimeStrategy)


def test_legacy_model_strategies_redirect_forecasting_models_to_runtime():
    params = OperationParameters(window_size=12, stride=1)

    regression_strategy = IndustrialSkLearnRegressionStrategy('lagged_ridge_forecaster', params=params)
    custom_regression_strategy = IndustrialCustomRegressionStrategy('okhs_fdmd_forecaster', params=params)

    assert isinstance(regression_strategy, IndustrialForecastingModelRuntimeStrategy)
    assert isinstance(custom_regression_strategy, IndustrialForecastingModelRuntimeStrategy)


def test_forecasting_runtime_redirect_helpers_centralize_boundary_policy():
    assert should_redirect_legacy_model_strategy(IndustrialSkLearnRegressionStrategy, 'lagged_ridge_forecaster')
    assert not should_redirect_legacy_model_strategy(IndustrialSkLearnRegressionStrategy, 'ridge')
    assert should_redirect_legacy_preprocessing_strategy(IndustrialPreprocessingStrategy, 'hankelisation')
    assert not should_redirect_legacy_preprocessing_strategy(IndustrialPreprocessingStrategy, 'scaling')


def test_legacy_forecasting_boundaries_now_use_shared_redirect_mixins():
    assert issubclass(IndustrialSkLearnEvaluationStrategy, LegacyForecastingModelRedirectMixin)
    assert issubclass(IndustrialCustomPreprocessingStrategy, LegacyForecastingPreprocessingRedirectMixin)


def test_neural_forecasting_strategy_is_extracted_to_dedicated_module():
    assert LegacyFedotNNTimeSeriesStrategy is ExtractedFedotNNTimeSeriesStrategy

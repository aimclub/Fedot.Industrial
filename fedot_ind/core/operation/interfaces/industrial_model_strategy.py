from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.time_series import FedotTsForecastingStrategy
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.operation.interfaces.detection_runtime_strategy import IndustrialDetectionModelRuntimeStrategy
from fedot_ind.core.operation.interfaces.neural_forecasting_strategy import FedotNNTimeSeriesStrategy  # noqa: F401
from fedot_ind.core.operation.interfaces.forecasting_runtime_strategy import (
    IndustrialForecastingModelRuntimeStrategy,
    LegacyForecastingModelRedirectMixin,
)
from fedot_ind.core.operation.interfaces.industrial_preprocessing_strategy import (
    IndustrialCustomPreprocessingStrategy, MultiDimPreprocessingStrategy)
from fedot_ind.core.repository.forecasting_registry import CANONICAL_STAGE_FORECASTING_MODELS
from fedot_ind.core.repository.model_repository import FORECASTING_MODELS, NEURAL_MODEL, SKLEARN_CLF_MODELS, \
    SKLEARN_REG_MODELS


class FedotNNClassificationStrategy(EvaluationStrategy):
    __operations_by_types = NEURAL_MODEL

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.output_mode = params.get('output_mode', 'labels')
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl,
                                                                  operation_type,
                                                                  mode='multi_dimensional')
        self.multi_dim_dispatcher.params_for_fit = params

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data)

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data, output_mode=output_mode)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data, output_mode=output_mode)


class FedotNNRegressionStrategy(FedotNNClassificationStrategy):
    __operations_by_types = NEURAL_MODEL

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.output_mode = params.get('output_mode', 'labels')
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(
            self.operation_impl, operation_type, mode='multi_dimensional')
        self.multi_dim_dispatcher.params_for_fit = params


class IndustrialSkLearnEvaluationStrategy(LegacyForecastingModelRedirectMixin, IndustrialCustomPreprocessingStrategy):

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.multi_dim_dispatcher.mode = 'one_dimensional'

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data)
        return self.multi_dim_dispatcher.fit(train_data)

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data, output_mode=output_mode)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data, output_mode=output_mode)


class IndustrialSkLearnClassificationStrategy(IndustrialSkLearnEvaluationStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """
    _operations_by_types = SKLEARN_CLF_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.multi_dim_dispatcher.mode = 'multi_dimensional' if self.operation_id.__contains__('industrial') \
            else self.multi_dim_dispatcher.mode


class IndustrialSkLearnRegressionStrategy(IndustrialSkLearnEvaluationStrategy):
    """ Strategy for applying regression algorithms from Sklearn library """
    _operations_by_types = SKLEARN_REG_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.multi_dim_dispatcher.mode = 'multi_dimensional' if self.operation_id.__contains__('industrial') \
            else self.multi_dim_dispatcher.mode

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data, output_mode='labels')

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        return self.multi_dim_dispatcher.predict_for_fit(
            trained_operation, predict_data, output_mode='labels')


class IndustrialSkLearnForecastingStrategy(IndustrialForecastingModelRuntimeStrategy):
    """Legacy compatibility wrapper over the forecasting runtime strategy."""

    _operations_by_types = FORECASTING_MODELS
    _runtime_operations = set(CANONICAL_STAGE_FORECASTING_MODELS) | {
        'eigen_forecaster',
        'glm',
        'lagged_forecaster',
        'patch_tst_model',
        'tst_model',
        'deepar_model',
        'tcn_model',
        'nbeats_model',
    }

    def __new__(cls, operation_type: str, params: Optional[OperationParameters] = None):
        if cls is IndustrialSkLearnForecastingStrategy and operation_type not in cls._runtime_operations:
            return FedotTsForecastingStrategy(operation_type, params)
        return super().__new__(cls)


class IndustrialCustomRegressionStrategy(IndustrialSkLearnEvaluationStrategy):
    _operations_by_types = SKLEARN_REG_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data)
        return self.multi_dim_dispatcher.fit(train_data)


class IndustrialAnomalyDetectionStrategy(IndustrialDetectionModelRuntimeStrategy):
    """Legacy compatibility wrapper over the detection runtime strategy."""

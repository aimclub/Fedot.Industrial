from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.time_series import FedotTsForecastingStrategy
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.nn.network_impl.patch_tst import PatchTSTModel
from fedot_ind.core.operation.interfaces.industrial_preprocessing_strategy import \
    (IndustrialCustomPreprocessingStrategy, MultiDimPreprocessingStrategy)
from fedot_ind.core.repository.model_repository import FORECASTING_MODELS, NEURAL_MODEL, SKLEARN_CLF_MODELS, \
    SKLEARN_REG_MODELS
from scipy.stats import kurtosis
from scipy.stats import skew


class FedotNNClassificationStrategy(EvaluationStrategy):
    __operations_by_types = NEURAL_MODEL

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')

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
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(self.operation_impl,
                                                                  operation_type,
                                                                  mode='multi_dimensional')
        self.multi_dim_dispatcher.params_for_fit = params


class FedotNNTimeSeriesStrategy(FedotTsForecastingStrategy):
    __operations_by_types = {
        'patch_tst_model': PatchTSTModel
    }

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)


class IndustrialSkLearnEvaluationStrategy(IndustrialCustomPreprocessingStrategy):

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


class IndustrialSkLearnRegressionStrategy(IndustrialSkLearnEvaluationStrategy):
    """ Strategy for applying regression algorithms from Sklearn library """
    _operations_by_types = SKLEARN_REG_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        return self.multi_dim_dispatcher.predict(trained_operation, predict_data, output_mode='labels')

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data)
        return self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data, output_mode='labels')


class IndustrialSkLearnForecastingStrategy(IndustrialSkLearnEvaluationStrategy):
    """ Strategy for applying forecasting algorithms """
    _operations_by_types = FORECASTING_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.multi_dim_dispatcher.mode = 'channel_independent'
        self.multi_dim_dispatcher.concat_func = np.vstack
        self.ensemble_func = np.sum

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data)
        return self.multi_dim_dispatcher.fit(train_data)

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        predict_data = self.multi_dim_dispatcher._convert_input_data(predict_data, mode=self.multi_dim_dispatcher.mode)
        predict_output = self.multi_dim_dispatcher.predict(trained_operation, predict_data, output_mode='labels')
        predict_output.predict = self.ensemble_func(predict_output.predict, axis=0)
        return predict_output

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'labels') -> OutputData:
        predict_data = self.multi_dim_dispatcher._convert_input_data(
            predict_data, mode=self.multi_dim_dispatcher.mode)
        predict_output = self.multi_dim_dispatcher.predict_for_fit(trained_operation, predict_data,
                                                                   output_mode='labels')
        predict_output.predict = self.ensemble_func(predict_output.predict, axis=0)
        return predict_output


class IndustrialCustomRegressionStrategy(IndustrialSkLearnEvaluationStrategy):
    _operations_by_types = SKLEARN_REG_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data)
        return self.multi_dim_dispatcher.fit(train_data)

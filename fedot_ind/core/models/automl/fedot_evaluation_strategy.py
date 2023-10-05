from typing import Optional

from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.automl.fedot_implementation import FedotClassificationImplementation, FedotRegressionImplementation


class FedotAutoMLClassificationStrategy(EvaluationStrategy):

    __operations_by_types = {
        'fedot_cls': FedotClassificationImplementation
    }

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        model = self.operation_impl(self.params_for_fit)
        model.fit(train_data)
        return model

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        n_classes = trained_operation.model.train_data.num_classes
        if self.output_mode == 'labels':
            prediction = trained_operation.model.predict(predict_data)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.model.predict_proba(predict_data)
            if n_classes < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            elif n_classes == 2 and self.output_mode != 'full_probs' and prediction.shape[1] > 1:
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted


class FedotAutoMLRegressionStrategy(EvaluationStrategy):

    __operations_by_types = {
        'fedot_regr': FedotRegressionImplementation
    }

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        model = self.operation_impl(self.params_for_fit)
        model.fit(train_data)
        return model

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        return trained_operation.model.predict(predict_data)



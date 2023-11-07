from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.nn.ts.inception import InceptionTimeModel


class FedotNNClassificationStrategy(EvaluationStrategy):
    __operations_by_types = {
        'inception_model': InceptionTimeModel
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
        return trained_operation.predict(predict_data, self.output_mode)


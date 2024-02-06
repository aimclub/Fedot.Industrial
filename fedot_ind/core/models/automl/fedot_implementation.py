from typing import Optional

from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation


class FedotClassificationImplementation(ModelImplementation):
    """Implementation of Fedot as classification pipeline node for AutoML.

    """
    AVAILABLE_OPERATIONS = default_industrial_availiable_operation(
        'classification')

    def __init__(self, params: Optional[OperationParameters] = None):
        if not params:
            params = OperationParameters()
        else:
            params = params.to_dict()
        if 'available_operations' not in params.keys():
            params.update({'available_operations': self.AVAILABLE_OPERATIONS})
        self.model = Fedot(**params)
        super(FedotClassificationImplementation, self).__init__()

    def fit(self, input_data: InputData):
        self.model.fit(input_data)
        return self

    def predict(self, input_data: InputData, output_mode='default') -> OutputData:
        return self.model.current_pipeline.predict(input_data, output_mode=output_mode)


class FedotRegressionImplementation(ModelImplementation):
    """Implementation of Fedot as regression pipeline node for AutoML.

    """
    AVAILABLE_OPERATIONS = default_industrial_availiable_operation(
        'regression')

    def __init__(self, params: Optional[OperationParameters] = None):
        if not params:
            params = OperationParameters()
        else:
            params = params.to_dict()
        if 'available_operations' not in params.keys():
            params.update({'available_operations': self.AVAILABLE_OPERATIONS})
        self.model = Fedot(**params)
        super(FedotRegressionImplementation, self).__init__()

    def fit(self, input_data: InputData):
        self.model.fit(input_data)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        return self.model.current_pipeline.predict(input_data)

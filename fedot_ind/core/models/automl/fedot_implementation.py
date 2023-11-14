from typing import Optional

from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.architecture.settings.constanst_repository import *


class FedotClassificationImplementation(ModelImplementation):
    AVAILABLE_OPERATIONS = AVAILABLE_CLS_OPERATIONS

    def __init__(self, params: Optional[OperationParameters] = None):
        if not params:
            params = OperationParameters()
        self.model = Fedot(problem='classification', available_operations=self.AVAILABLE_OPERATIONS, **params.to_dict())
        super(FedotClassificationImplementation, self).__init__()

    def fit(self, input_data: InputData):
        self.model.fit(input_data)
        return self

    def predict(self, input_data: InputData, output_mode='default') -> OutputData:
        return self.model.current_pipeline.predict(input_data, output_mode=output_mode)


class FedotRegressionImplementation(ModelImplementation):
    AVAILABLE_OPERATIONS = AVAILABLE_REG_OPERATIONS
    def __init__(self, params: Optional[OperationParameters] = None):
        if not params:
            params = OperationParameters()
        self.model = Fedot(problem='regression', available_operations=self.AVAILABLE_OPERATIONS,
                           **params.to_dict())
        super(FedotRegressionImplementation, self).__init__()

    def fit(self, input_data: InputData):
        self.model.fit(input_data)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        return self.model.current_pipeline.predict(input_data)

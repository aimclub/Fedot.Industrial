from typing import Optional

from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FedotClassificationImplementation(ModelImplementation):
    AVAILABLE_OPERATIONS = ['rf',
                            'logit',
                            'scaling',
                            'normalization',
                            'pca',
                            'catboost',
                            'svc',
                            'knn',
                            'fast_ica',
                            'kernel_pca',
                            'isolation_forest_class']

    def __init__(self, params: Optional[OperationParameters] = None):
        if not params:
            params = OperationParameters()
        self.model = Fedot(problem='classification', available_operations=self.AVAILABLE_OPERATIONS, **params.to_dict())
        super(FedotClassificationImplementation, self).__init__()

    def fit(self, input_data: InputData):
        self.model.fit(input_data)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        return self.model.current_pipeline.predict(input_data)


class FedotRegressionImplementation(ModelImplementation):
    AVAILABLE_OPERATIONS = ['rfr',
                            'ridge',
                            'scaling',
                            'normalization',
                            'pca',
                            'catboostreg',
                            'xgbreg',
                            'svr',
                            'dtreg',
                            'treg',
                            'knnreg',
                            'fast_ica',
                            'kernel_pca',
                            'isolation_forest_reg',
                            'rfe_lin_reg',
                            'rfe_non_lin_reg']

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

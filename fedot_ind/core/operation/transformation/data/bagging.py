from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum

from fedot_ind.core.repository.constanst_repository import BAGGING_METHOD


class BaggingEnsemble(ModelImplementation):
    """Class ensemble predictions.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.bagging_method = params.get('method', 'weighted')
        self.stride = params.get('stride', 1)
        self.add_global_features = params.get('add_global_features', True)
        self.is_linreg_ensemble = self.bagging_method.__contains__('weighted')

    def __repr__(self):
        return 'Statistical Class for TS representation'

    def fit(self, input_data: InputData):
        """ Method fit model on a dataset

        :param input_data: data with features, target and ids to process
        """
        if input_data.task.task_type == TaskTypesEnum.ts_forecasting and len(input_data.target.shape) > 1:
            input_data.target = input_data.target[-1]
        if self.is_linreg_ensemble:
            self.method_impl = BAGGING_METHOD[self.bagging_method]()
            self.method_impl.fit(input_data.features, input_data.target)
        else:
            self.method_impl = BAGGING_METHOD[self.bagging_method]
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """ Method make prediction

        :param input_data: data with features, target and ids to process
        """
        if not self.is_linreg_ensemble:
            ensembled_predict = self.method_impl(input_data.features, axis=1)
        else:
            ensembled_predict = self.method_impl.predict(input_data.features)
        return ensembled_predict

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        """ Method make prediction while graph fitting.
        Allows to implement predict method different from main predict method
        if another behaviour for fit graph stage is needed.

        :param input_data: data with features, target and ids to process
        """
        return self.predict(input_data)

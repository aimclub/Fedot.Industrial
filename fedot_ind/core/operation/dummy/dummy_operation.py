from typing import Optional

import pandas as pd
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.architecture.settings.computational import backend_methods as np


def check_multivariate_data(data: pd.DataFrame) -> tuple:
    """
    Checks if the provided pandas DataFrame contains multivariate data.

    Args:
        data (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
        bool: True if the DataFrame contains multivariate data (nested columns), False otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        return len(data.shape) > 2, data
    else:
        return isinstance(data.iloc[0, 0], pd.Series), data.values


class DummyOperation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.average = params.get('average_type', None)
        self.prediction_length = params.get('prediction_length', None)

    def fit(self, input_data: InputData):
        pass

    def transform(self, input_data: InputData) -> OutputData:
        if self.average is not None:
            transformed_features = np.average(
                input_data.features.reshape(-1, self.prediction_length), axis=0)
        else:
            transformed_features = input_data.features
        predict = self._convert_to_output(
            input_data, transformed_features, data_type=DataTypesEnum.table)
        return predict

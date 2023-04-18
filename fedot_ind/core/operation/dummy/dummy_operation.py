from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.repository.dataset_types import DataTypesEnum


class DummyOperation(DataOperationImplementation):

    def fit(self, input_data: InputData):
        pass

    def transform(self, input_data: InputData) -> OutputData:
        predict = self._convert_to_output(input_data, input_data.features, data_type=DataTypesEnum.table)
        return predict

import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.time_series import FedotTsForecastingStrategy
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.nn.network_impl.forecasting_model.deep_tcn import TCNModel
from fedot_ind.core.models.nn.network_impl.forecasting_model.deepar import DeepAR
from fedot_ind.core.models.nn.network_impl.forecasting_model.nbeats import NBeatsModel
from fedot_ind.core.models.nn.network_impl.forecasting_model.patch_tst import PatchTSTModel


class FedotNNTimeSeriesStrategy(FedotTsForecastingStrategy):
    """Forecasting-oriented neural strategy extracted from legacy industrial model internals."""

    __operations_by_types = {
        'patch_tst_model': PatchTSTModel,
        'nbeats_model': NBeatsModel,
        'deepar_model': DeepAR,
        'tcn_model': TCNModel,
    }

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = self.operation(self.params_for_fit)
        model.fit(train_data)
        return model

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        prediction = trained_operation.predict(predict_data, output_mode)
        return self._convert_to_output(prediction, predict_data)

    def predict_for_fit(self, trained_operation, predict_data: InputData,
                        output_mode: str = 'default') -> OutputData:
        prediction = trained_operation.predict_for_fit(predict_data, output_mode)
        return self._convert_to_output(prediction, predict_data)

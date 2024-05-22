from copy import copy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot_ind.core.models.ts_forecasting.markov_extension import MSARExtension

from sklearn.preprocessing import StandardScaler

class MarkovAR(ModelImplementation):

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.autoreg = None
        self.actual_ts_len = None
        self.scaler = StandardScaler()

    def fit(self, input_data):
        """ Class fit ar model on data

        :param input_data: data with features, target and ids to process
        """

        source_ts = pd.DataFrame(input_data.features)
        source_ts = pd.DataFrame(self.scaler.fit_transform(input_data.features.reshape(-1, 1)))
        
        self.actual_ts_len = len(source_ts)

        self.autoreg = MarkovAutoregression(source_ts, 
                        k_regimes=2, 
                        order=4,
                        switching_variance=True).fit()
        
        self.actual_ts_len = input_data.idx.shape[0]

        return self.autoreg

    def predict(self, input_data):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :return output_data: output data with smoothed time series
        """
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length

        # in case in(out) sample forecasting
        self.handle_new_data(input_data)
        start_id = self.actual_ts_len
        end_id = start_id + forecast_length - 1
        # predicted = self.autoreg.predict(start=start_id, end=end_id)

        predicted = MSARExtension(self.autoreg).predict_out_of_sample()
        # print('Pred. shaper', predicted.shape)

        predict = self.scaler.inverse_transform(np.array([predicted]).ravel().reshape(1, -1))
        # predict = np.array(predicted).reshape(1, -1)

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        idx = input_data.idx
        target = input_data.target
        predicted = self.autoreg.predict(start=idx[0], end=idx[-1])
        # adding nan to target as in predicted
        nan_mask = np.isnan(predicted)
        target = target.astype(float)
        target = target[~nan_mask]
        idx = idx[~nan_mask]
        predicted = predicted[~nan_mask]
        new_idx, predict = ts_to_table(idx=idx,
                                       time_series=predicted,
                                       window_size=forecast_length)
        _, target_columns = ts_to_table(idx=idx,
                                        time_series=target,
                                        window_size=forecast_length)
        input_data.idx = new_idx
        input_data.target = target_columns
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def handle_new_data(self, input_data: InputData):
        """
        Method to update x samples inside a model (used when we want to use old model to a new data)

        :param input_data: new input_data
        """
        if input_data.idx[0] > self.actual_ts_len:
            self.autoreg.model.endog = input_data.features[-self.actual_ts_len:]
            self.autoreg.model._setup_regressors()


class ExpSmoothingImplementation(ModelImplementation):
    """ Exponential smoothing implementation from statsmodels """

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.model = None
        if self.params.get("seasonal"):
            self.seasonal_periods = int(self.params.get("seasonal_periods"))
        else:
            self.seasonal_periods = None

    def fit(self, input_data):
        self.model = ETSModel(
            input_data.features.astype("float64"),
            error=self.params.get("error"),
            trend=self.params.get("trend"),
            seasonal=self.params.get("seasonal"),
            damped_trend=self.params.get("damped_trend") if self.params.get("trend") else None,
            seasonal_periods=self.seasonal_periods
        )
        self.model = self.model.fit(disp=False)
        return self.model

    def predict(self, input_data):
        input_data = copy(input_data)
        idx = input_data.idx

        start_id = idx[0]
        end_id = idx[-1]
        predictions = self.model.predict(start=start_id,
                                         end=end_id)
        predict = predictions
        predict = np.array(predict).reshape(1, -1)
        new_idx = np.arange(start_id, end_id + 1)

        input_data.idx = new_idx

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        idx = input_data.idx
        target = input_data.target

        # Indexing for statsmodels is different
        start_id = idx[0]
        end_id = idx[-1]
        predictions = self.model.predict(start=start_id,
                                         end=end_id)
        _, predict = ts_to_table(idx=idx,
                                 time_series=predictions,
                                 window_size=forecast_length)
        new_idx, target_columns = ts_to_table(idx=idx,
                                              time_series=target,
                                              window_size=forecast_length)

        input_data.idx = new_idx
        input_data.target = target_columns

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data
from abc import abstractmethod

import pandas as pd

from typing import Optional
from sklearn.preprocessing import StandardScaler
from torch import cuda, device

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly_detector import AnomalyDetector

device = device("cuda:0" if cuda.is_available() else "cpu")


class AutoEncoderDetector(AnomalyDetector):
    """A reconstruction autoencoder model to detect anomalies
    in timeseries data using reconstruction error as an anomaly score.


    Args:
        params: additional parameters for an encapsulated autoencoder model

            .. details:: Possible parameters:

                    - ``learning_rate`` -> learning rate for an optimizer
                    - ``ucl_quantile`` -> upper control limit quantile
                    - ``n_steps_share`` -> share of an n_steps to define n_steps window
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.learning_rate = self.params.get('learning_rate', 0.001)
        self.ucl_quantile = self.params.get('ucl_quantile', 0.999)
        self.n_steps_share = self.params.get('n_steps_share', 0.15)
        self.transformation_mode = 'full'
        self.scaler = StandardScaler()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    def fit(self, input_data: InputData) -> None:
        self.n_steps = round(input_data.features.shape[0] * self.n_steps_share)
        converted_input_data = self.convert_input_data(input_data)
        self.model_impl = self.build_model()

        self.model_impl.fit(converted_input_data)
        self.ucl = pd.Series(
            np.abs(converted_input_data - self.model_impl.predict(converted_input_data)).mean(axis=1).sum(axis=1)
        ).quantile(self.ucl_quantile) * 4 / 3

    def predict(self, input_data: InputData):
        converted_input_data = self.convert_input_data(input_data, fit_stage=False)
        prediction = np.zeros(input_data.target.shape)

        cnn_residuals = pd.Series(
            np.abs(converted_input_data - self.model_impl.predict(converted_input_data)).mean(axis=1).sum(axis=1)
        )
        anomalous_data = cnn_residuals > self.ucl
        anomalous_data_indices = []
        # data i is an anomaly if samples [(i - n_steps + 1) to (i)] are anomalies
        for data_idx in range(self.n_steps - 1, len(converted_input_data) - self.n_steps + 1):
            if np.all(anomalous_data[data_idx - self.n_steps + 1: data_idx]):
                anomalous_data_indices.append(data_idx)

        labels = pd.Series(data=0)
        labels.iloc[anomalous_data_indices] = 1
        labels = labels.values.reshape(-1, 1)
        start_idx, end_idx = prediction.shape[0] - labels.shape[0], prediction.shape[0]
        prediction[np.arange(start_idx, end_idx), :] = labels
        return prediction

    def convert_input_data(self, input_data: InputData, fit_stage: bool = True) -> np.ndarray:
        if fit_stage:
            values = self.scaler.fit_transform(input_data.features)
        else:
            values = self.scaler.transform(input_data.features)
        output = []
        for i in range(len(values) - self.n_steps + 1):
            output.append(values[i: (i + self.n_steps)])
        return np.stack(output)

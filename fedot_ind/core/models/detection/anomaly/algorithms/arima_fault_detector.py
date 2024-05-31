import pandas as pd
from arimafd import Arima_anomaly_detection
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly_detector import AnomalyDetector


class ARIMAFaultDetector(AnomalyDetector):
    """
    ARIMA fault detection algorithm. The idea behind this is to use ARIMA weights
    as features for the anomaly detection algorithm. Using discrete differences of weight coefficients
    for different heuristic methods for obtaining function, which characterized the state using a threshold.

    arimafd dependency:
    pip install -U git+https://github.com/Lopa10ko/arimafd.git@fix-sklearn
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.ar_order = self.params.get('ar_order', 3)
        self.transformation_mode = 'full'
        self.transformation_type = pd.DataFrame

    def build_model(self):
        return Arima_anomaly_detection(ar_order=self.ar_order)

    def predict(self, input_array: InputData) -> np.ndarray:
        converted_input_data = self.convert_input_data(input_array, fit_stage=False)
        prediction = np.zeros(input_array.target.shape)
        labels = self.model_impl.predict(converted_input_data).values.reshape(-1, 1)
        start_idx, end_idx = prediction.shape[0] - labels.shape[0], prediction.shape[0]
        prediction[np.arange(start_idx, end_idx), :] = labels
        return prediction

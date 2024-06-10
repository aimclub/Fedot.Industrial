from typing import Optional

import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.ensemble import IsolationForest as SklearnIsolationForest

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly_detector import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest or iForest builds an ensemble of iTrees for a given data set, then anomalies are those
    instances which have short average path lengths on the iTrees.

    Args:
        params: additional parameters for a IsolationForest model

            .. details:: Possible parameters:

                    - ``random_state`` -> random seed used for reproducibility
                    - ``n_jobs`` -> number of CPU cores to use for parallelism
                    - ``contamination`` -> expected proportion of anomalies in the dataset
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.random_state = self.params.get('random_state', 0)
        self.n_jobs = self.params.get('n_jobs', -1)
        self.contamination = self.params.get('contamination', 0.0005)
        self.transformation_mode = 'full'

    def build_model(self):
        return SklearnIsolationForest(random_state=self.random_state,
                                      n_jobs=self.n_jobs,
                                      contamination=self.contamination)

    def predict(self, input_data: InputData) -> np.ndarray:
        converted_input_data = self.convert_input_data(input_data, fit_stage=False)
        prediction = np.zeros(input_data.target.shape)
        labels = pd.Series(self.model_impl.predict(converted_input_data) * (-1)) \
            .rolling(3).median().fillna(0).replace(-1, 0).values.reshape(-1, 1)
        start_idx, end_idx = prediction.shape[0] - labels.shape[0], prediction.shape[0]
        prediction[np.arange(start_idx, end_idx), :] = labels
        return prediction

    def predict_for_fit(self, input_array: InputData):
        converted_input_data = self.convert_input_data(input_array, fit_stage=False)
        prediction = np.zeros(input_array.target.shape)
        labels = pd.Series(self.model_impl.predict(converted_input_data) * (-1)) \
            .rolling(3).median().fillna(0).replace(-1, 0).values.reshape(-1, 1)
        start_idx, end_idx = prediction.shape[0] - labels.shape[0], prediction.shape[0]
        prediction[np.arange(start_idx, end_idx), :] = labels
        return prediction

from abc import abstractmethod
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.utilities.custom_errors import AbstractMethodNotImplementError

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.repository.constanst_repository import FEDOT_TASK


class AnomalyDetector(ModelImplementation):

    def __init__(self, params: Optional[OperationParameters] = None) -> None:
        super().__init__(params)
        self.length_of_detection_window = self.params.get('window_length', 10)
        self.anomaly_threshold = self.params.get('anomaly_thr', 0.9)

    @property
    def classes_(self) -> int:
        return 1

    def convert_input_data(
            self,
            input_data: InputData,
            fit_stage: bool = True) -> InputData:
        feature_matrix = np.concatenate(
            [
                HankelMatrix(
                    time_series=ts,
                    window_size=self.window_size
                ).trajectory_matrix.T for ts in input_data.features.T
            ],
            axis=1
        )
        if fit_stage:  # shrink target
            target = input_data.target[:feature_matrix.shape[0]]
        else:  # augmented predict
            target = input_data.target

        converted_input_data = InputData(
            idx=np.arange(feature_matrix.shape[0]),
            features=feature_matrix,
            target=target,
            task=FEDOT_TASK['anomaly_detection'],
            data_type=DataTypesEnum.table
        )
        converted_input_data.supplementary_data.is_auto_preprocessed = True
        return self.convert_data_for_model(
            converted_input_data, fit_stage=fit_stage)

    @abstractmethod
    def build_model(self):
        raise AbstractMethodNotImplementError

    @staticmethod
    @abstractmethod
    def convert_data_for_model(input_data: InputData, fit_stage: bool = True):
        raise AbstractMethodNotImplementError

    def fit(self, input_data: InputData) -> None:
        self.model_impl = self.build_model()
        self.window_size = round(
            input_data.features.shape[0] * (self.length_of_detection_window / 100))
        converted_input_data = self.convert_input_data(input_data)
        self.model_impl.fit(converted_input_data)

    def predict(self, input_data: InputData) -> np.ndarray:
        converted_input_data = self.convert_input_data(
            input_data, fit_stage=False)
        probs = self.model_impl.predict(converted_input_data).predict
        # TODO: use a broader interface for predict calls
        labels = np.apply_along_axis(
            self.convert_probs_to_labels, 1, probs).reshape(-1, 1)
        prediction = np.zeros(converted_input_data.target.shape)
        start_idx, end_idx = prediction.shape[0] - \
            labels.shape[0], prediction.shape[0]
        prediction[np.arange(start_idx, end_idx), :] = labels
        return prediction

    def predict_proba(self, input_data: InputData) -> np.ndarray:
        converted_input_data = self.convert_input_data(
            input_data, fit_stage=False)
        probs = self.model_impl.predict(converted_input_data).predict
        prediction = np.zeros(
            (converted_input_data.target.shape[0], probs.shape[1]))
        start_idx, end_idx = prediction.shape[0] - \
            probs.shape[0], prediction.shape[0]
        prediction[np.arange(start_idx, end_idx), :] = probs
        return prediction

    def convert_probs_to_labels(self, prob_matrix_row) -> int:
        return 1 if prob_matrix_row[1] > self.anomaly_threshold else 0

    def score_samples(self, input_data: InputData) -> np.ndarray:
        return self.predict_for_fit(input_data).predict

    def predict_for_fit(self, input_data: InputData):
        return self.model_impl.predict(self.convert_input_data(input_data))

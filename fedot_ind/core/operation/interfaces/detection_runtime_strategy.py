from __future__ import annotations

import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler

from fedot_ind.core.models.detection.anomaly.algorithms.arima_fault_detector import ARIMAFaultDetector
from fedot_ind.core.models.detection.anomaly.algorithms.lstm_autoencoder_detector import LSTMAutoEncoderDetector
from fedot_ind.core.models.detection.modern_detectors import (
    ConvAutoencoderDetector,
    FeatureIsolationForestDetector,
    FeatureOneClassDetector,
    TCNAutoencoderDetector,
)
from fedot_ind.core.models.detection.runtime import DetectionBoundaryAdapter, ensure_detection_array
from fedot_ind.core.repository.detection_registry import canonical_detection_model_name

DETECTION_RUNTIME_MODELS = {
    'feature_iforest_detector': FeatureIsolationForestDetector,
    'feature_oneclass_detector': FeatureOneClassDetector,
    'conv_autoencoder_detector': ConvAutoencoderDetector,
    'tcn_autoencoder_detector': TCNAutoencoderDetector,
    'legacy_lstm_autoencoder_detector': LSTMAutoEncoderDetector,
    'legacy_arima_detector': ARIMAFaultDetector,
}


def build_detection_boundary_batch(input_data: InputData, params: Optional[OperationParameters] = None):
    """Build a FEDOT-boundary detection batch using the shared runtime adapter."""
    params = params or OperationParameters()
    return DetectionBoundaryAdapter.from_input_data(
        input_data,
        window_size=params.get('window_size'),
        window_size_percent=params.get('window_size_percent', params.get('window_length')),
        stride=params.get('stride'),
        metadata={'runtime': 'industrial_detection'},
    )


class IndustrialDetectionModelRuntimeStrategy(EvaluationStrategy):
    """FEDOT evaluation strategy for modern and legacy anomaly detectors."""

    _operations_by_types = DETECTION_RUNTIME_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.canonical_operation_type = canonical_detection_model_name(operation_type)
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.output_mode = self.params_for_fit.get('output_mode', 'labels')

    def _convert_to_operation(self, operation_type: str):
        canonical_name = canonical_detection_model_name(operation_type)
        if canonical_name in self._operations_by_types:
            return self._operations_by_types[canonical_name]
        raise ValueError(f'Impossible to obtain detection runtime strategy for {operation_type}')

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        train_data = self._normalize_input_data(train_data)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self._build_output(trained_operation, predict_data, output_mode)

    def predict_for_fit(self, trained_operation, predict_data: InputData,
                        output_mode: str = 'default') -> OutputData:
        fit_output_mode = 'probs' if output_mode == 'default' else output_mode
        return self._build_output(trained_operation, predict_data, fit_output_mode)

    def _build_output(self, trained_operation, predict_data: InputData, output_mode: str) -> OutputData:
        predict_data = self._normalize_input_data(predict_data)
        resolved_mode = output_mode if output_mode != 'default' else self.output_mode
        if resolved_mode == 'default':
            resolved_mode = 'labels'
        if not hasattr(trained_operation, 'score_series_on_values'):
            prediction = self._predict_by_mode(trained_operation, predict_data, resolved_mode)
            return self._convert_to_output(prediction, predict_data, output_data_type=DataTypesEnum.table)

        score_series = trained_operation.score_series_on_values(predict_data)
        return DetectionBoundaryAdapter.to_output_data(
            predict_data,
            score_series=score_series,
            predict_mode=resolved_mode,
        )

    def _predict_by_mode(self, trained_operation, input_data: InputData, output_mode: str):
        resolved_mode = self.output_mode if output_mode == 'default' else output_mode
        if resolved_mode in {'probs', 'probabilities'}:
            return trained_operation.predict_proba(input_data)
        if resolved_mode in {'scores', 'score'}:
            return trained_operation.score_samples(input_data)
        return trained_operation.predict(input_data)

    def _normalize_input_data(self, input_data: InputData) -> InputData:
        features = ensure_detection_array(input_data.features)
        if getattr(input_data.features, 'ndim', 1) <= 2:
            return input_data
        return InputData(
            idx=input_data.idx,
            features=features,
            target=input_data.target,
            task=input_data.task,
            data_type=input_data.data_type,
        )

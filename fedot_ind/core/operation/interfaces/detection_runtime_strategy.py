from __future__ import annotations

import warnings
from inspect import signature
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler

from fedot_ind.core.models.detection.runtime import DetectionBoundaryAdapter, DetectionWindowBatch
from fedot_ind.core.repository.model_repository import ANOMALY_DETECTION_MODELS


def is_detection_model_operation(operation_type: str) -> bool:
    return str(operation_type) in ANOMALY_DETECTION_MODELS


def _supports_output_mode(method) -> bool:
    try:
        return 'output_mode' in signature(method).parameters
    except (TypeError, ValueError):
        return False


def _invoke_detection_method(method, input_data: InputData, *, output_mode: str):
    if _supports_output_mode(method):
        return method(input_data, output_mode=output_mode)
    return method(input_data)


def build_detection_boundary_batch(
        input_data: InputData,
        params: Optional[OperationParameters] = None,
) -> DetectionWindowBatch:
    params = params or OperationParameters()
    return DetectionBoundaryAdapter.from_input_data(
        input_data,
        window_size=params.get('window_size'),
        window_size_percent=params.get('window_size_percent', params.get('window_length')),
        stride=params.get('stride'),
        metadata={
            'problem': 'anomaly_detection',
            'operation_type': getattr(input_data, 'task', None),
        },
    )


def _invoke_detection_prediction(
        trained_operation,
        predict_data: InputData,
        *,
        output_mode: str,
):
    normalized_mode = str(output_mode or 'default').lower()
    if normalized_mode in {'default', 'probs', 'probabilities'}:
        if hasattr(trained_operation, 'predict_proba'):
            return trained_operation.predict_proba(predict_data)
        if hasattr(trained_operation, 'score_samples'):
            return trained_operation.score_samples(predict_data)
        return _invoke_detection_method(trained_operation.predict, predict_data, output_mode=output_mode)
    if normalized_mode in {'scores', 'score_samples'} and hasattr(trained_operation, 'score_samples'):
        return trained_operation.score_samples(predict_data)
    return _invoke_detection_method(trained_operation.predict, predict_data, output_mode=output_mode)


class IndustrialDetectionModelRuntimeStrategy(EvaluationStrategy):
    """Thin anomaly-detection runtime strategy without classification-style dispatch."""

    _operations_by_types = ANOMALY_DETECTION_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.output_mode = self.params_for_fit.get('output_mode', 'default')

    def _instantiate(self):
        return self.operation_impl(self.params_for_fit)

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self._instantiate()
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        prediction = _invoke_detection_prediction(
            trained_operation,
            predict_data,
            output_mode=output_mode,
        )
        return self._convert_to_output(prediction, predict_data, output_data_type=DataTypesEnum.table)

    def predict_for_fit(self, trained_operation, predict_data: InputData,
                        output_mode: str = 'default') -> OutputData:
        if hasattr(trained_operation, 'predict_for_fit'):
            prediction = _invoke_detection_method(
                trained_operation.predict_for_fit,
                predict_data,
                output_mode=output_mode,
            )
        else:
            prediction = _invoke_detection_prediction(
                trained_operation,
                predict_data,
                output_mode=output_mode,
            )
        return self._convert_to_output(prediction, predict_data, output_data_type=DataTypesEnum.table)

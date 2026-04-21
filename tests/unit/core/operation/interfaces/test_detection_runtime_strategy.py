import numpy as np
import pytest

fedot_data = pytest.importorskip('fedot.core.data.data')
InputData = fedot_data.InputData
OutputData = fedot_data.OutputData
OperationParameters = pytest.importorskip('fedot.core.operations.operation_parameters').OperationParameters
DataTypesEnum = pytest.importorskip('fedot.core.repository.dataset_types').DataTypesEnum
tasks_module = pytest.importorskip('fedot.core.repository.tasks')
Task = tasks_module.Task
TaskTypesEnum = tasks_module.TaskTypesEnum

from fedot_ind.core.operation.interfaces.detection_runtime_strategy import (
    IndustrialDetectionModelRuntimeStrategy,
    build_detection_boundary_batch,
)
from fedot_ind.core.operation.interfaces.industrial_model_strategy import IndustrialAnomalyDetectionStrategy


def _build_detection_input(length: int = 80) -> InputData:
    time = np.arange(length, dtype=float)
    series = np.sin(time / 4.0) + 0.05 * np.cos(time / 7.0)
    series[52:58] += 4.0
    features = series.reshape(-1, 1)
    return InputData(
        idx=np.arange(length),
        features=features,
        target=np.zeros(length, dtype=int),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_detection_runtime_strategy_feature_iforest_returns_output_data_for_labels_and_probs():
    input_data = _build_detection_input()
    params = OperationParameters(window_length=12, threshold_quantile=0.95, random_state=42, n_estimators=50)
    strategy = IndustrialDetectionModelRuntimeStrategy('feature_iforest_detector', params=params)

    trained_operation = strategy.fit(input_data)
    labels = strategy.predict(trained_operation, input_data, output_mode='labels')
    probs = strategy.predict(trained_operation, input_data, output_mode='probs')
    fit_scores = strategy.predict_for_fit(trained_operation, input_data, output_mode='probs')

    assert isinstance(labels, OutputData)
    assert isinstance(probs, OutputData)
    assert isinstance(fit_scores, OutputData)
    assert labels.data_type is DataTypesEnum.table
    assert np.asarray(labels.predict).shape == (80, 1)
    assert np.asarray(probs.predict).shape == (80, 2)
    assert np.asarray(fit_scores.predict).shape == (80, 2)


def test_build_detection_boundary_batch_respects_window_length_parameter():
    input_data = _build_detection_input(length=48)
    params = OperationParameters(window_length=25)

    batch = build_detection_boundary_batch(input_data, params=params)

    assert batch.window_size == 12
    assert batch.original_length == 48


def test_legacy_detection_strategy_is_now_a_runtime_wrapper():
    assert issubclass(IndustrialAnomalyDetectionStrategy, IndustrialDetectionModelRuntimeStrategy)

import json
from pathlib import Path

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.data.hankelisation import (
    HankelisationImplementation,
    normalize_hankelisation_params,
)
from fedot_ind.core.repository.constanst_repository import IND_DATA_OPERATION_PATH


def _build_ts_input(horizon: int = 14):
    series = np.arange(160, dtype=float)
    return InputData(
        idx=np.arange(len(series)),
        features=series.reshape(-1, 1),
        target=series,
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon)),
        data_type=DataTypesEnum.ts,
    )


def test_normalize_hankelisation_params_clips_to_valid_bounds():
    window_size, stride = normalize_hankelisation_params(
        time_series_length=40,
        forecast_length=14,
        window_size=200,
        stride=30,
    )

    assert window_size == 26
    assert stride == 13


def test_hankelisation_transform_for_fit_matches_reference_contract():
    input_data = _build_ts_input()
    operation = HankelisationImplementation({'window_size': 16, 'stride': 2})
    operation.fit(input_data)
    transformed = operation.transform_for_fit(input_data)

    reference_features = HankelMatrix(
        time_series=input_data.features,
        window_size=16,
        strides=2,
    ).trajectory_matrix.T
    reference_target = HankelMatrix(
        time_series=np.ravel(input_data.features)[16:],
        window_size=input_data.task.task_params.forecast_length + 1,
        strides=2,
    ).trajectory_matrix.T
    reference_features = reference_features[:reference_target.shape[0], :]

    assert transformed.data_type is DataTypesEnum.table
    assert np.array_equal(transformed.features, reference_features)
    assert np.array_equal(transformed.predict, reference_features)
    assert np.array_equal(transformed.target, reference_target)
    assert transformed.idx.shape[0] == reference_features.shape[0]


def test_hankelisation_transform_uses_same_stride_for_inference_alignment():
    input_data = _build_ts_input()
    operation = HankelisationImplementation({'window_size': 16, 'stride': 3})
    operation.fit(input_data)
    transformed = operation.transform(input_data)

    reference_features = HankelMatrix(
        time_series=input_data.features,
        window_size=16,
        strides=3,
    ).trajectory_matrix.T

    assert transformed.data_type is DataTypesEnum.table
    assert transformed.predict.shape[0] == 1
    assert transformed.predict.shape[1] == reference_features.shape[1]
    assert np.array_equal(transformed.features, reference_features[-1:, :])
    assert np.array_equal(transformed.predict, reference_features[-1:, :])
    assert transformed.target.shape[0] == 1
    assert transformed.idx.shape[0] == 1


def test_hankelisation_repository_metadata_supports_regression_tuning():
    repository_path = Path(IND_DATA_OPERATION_PATH)
    payload = json.loads(repository_path.read_text())

    meta_name = payload['operations']['hankelisation']['meta']
    meta_tasks = payload['metadata'][meta_name]['tasks']

    assert meta_name == 'hankelisation_transformation'
    assert 'TaskTypesEnum.regression' in meta_tasks
    assert 'TaskTypesEnum.ts_forecasting' in meta_tasks

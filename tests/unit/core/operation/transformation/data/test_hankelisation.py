import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, strategies as st
from fedot.core.data.input_data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

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

    series = input_data.features.reshape(-1)
    starts = range(0, len(series) - 16 - 14 + 1, 2)
    reference_features = np.array([series[start:start + 16] for start in starts])
    reference_target = np.array([series[start + 16:start + 30] for start in starts])

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

    reference_features = input_data.features.reshape(-1)[-16:].reshape(1, -1)

    assert transformed.data_type is DataTypesEnum.table
    assert transformed.predict.shape[0] == 1
    assert transformed.predict.shape[1] == reference_features.shape[1]
    assert np.array_equal(transformed.features, reference_features)
    assert np.array_equal(transformed.predict, reference_features)
    assert transformed.target is None
    assert transformed.idx.shape[0] == 1


@given(length=st.integers(12, 80), horizon=st.integers(1, 10),
       window=st.integers(2, 20), stride=st.integers(1, 8))
def test_hankel_windows_match_independent_slices_without_overlap(length, horizon, window, stride):
    data = _build_ts_input(horizon)
    data.features = np.arange(length, dtype=float)
    data.target = data.features.copy()
    data.idx = np.arange(length) + 100
    snapshot = data.features.copy()
    model = HankelisationImplementation({'window_size': window, 'stride': stride}).fit(data)
    output = model.transform_for_fit(data)
    width, step = normalize_hankelisation_params(length, horizon, window, stride)
    starts = range(0, length - width - horizon + 1, step)
    np.testing.assert_array_equal(output.features, [snapshot[i:i + width] for i in starts])
    np.testing.assert_array_equal(output.target, [snapshot[i + width:i + width + horizon] for i in starts])
    np.testing.assert_array_equal(output.idx, [data.idx[i + width] for i in starts])
    np.testing.assert_array_equal(data.features, snapshot)
    assert output.features.shape[1] == width
    assert output.target.shape[1] == horizon
    # Forecasting uses the final fitted-width state, even when stride does not divide the history length.
    np.testing.assert_array_equal(model.transform(data).features, snapshot[-width:].reshape(1, -1))
    assert model.transform(data).target is None


@pytest.mark.parametrize('horizon, width, length', [(1, 2, 3), (10, 2, 12), (10, 2, 30)])
def test_hankel_boundary_windows_do_not_resize_into_future(horizon, width, length):
    data = _build_ts_input(horizon)
    data.features, data.target, data.idx = np.arange(length), np.arange(length), np.arange(length)
    model = HankelisationImplementation({'window_size': width}).fit(data)
    output = model.transform_for_fit(data)
    assert output.features.shape == (length - width - horizon + 1, width)
    assert output.target.shape == (length - width - horizon + 1, horizon)
    np.testing.assert_array_equal(output.features[0], np.arange(width))
    np.testing.assert_array_equal(output.target[0], np.arange(width, width + horizon))
    data.features = data.features[-width:]
    data.idx = np.arange(length, length + horizon)
    np.testing.assert_array_equal(model.transform(data).features, np.arange(length - width, length)[None, :])


@pytest.mark.parametrize('features', [np.ones((2, 20)), np.array([1., np.nan, 2.]), np.arange(3)])
def test_hankel_rejects_unsupported_or_insufficient_history(features):
    data = _build_ts_input(3)
    data.features = features
    with pytest.raises(ValueError):
        HankelisationImplementation({'window_size': 2}).fit(data)


def test_hankelisation_repository_metadata_supports_regression_tuning():
    repository_path = Path(IND_DATA_OPERATION_PATH)
    payload = json.loads(repository_path.read_text())

    meta_name = payload['operations']['hankelisation']['meta']
    meta_tasks = payload['metadata'][meta_name]['tasks']

    assert meta_name == 'hankelisation_transformation'
    assert 'TaskTypesEnum.regression' in meta_tasks
    assert 'TaskTypesEnum.ts_forecasting' in meta_tasks

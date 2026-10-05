import json

import numpy as np
import pandas as pd
import pytest

from fedot.core.data.input_data.data import InputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.integration.fedot import (
    AxisLayout,
    DataProfile,
    DataStage,
    IntegrationContractError,
    IntegrationErrorCode,
    IntegrationTask,
    PredictionBatch,
    build_data_plan,
    normalize_input_data,
    validate_prediction_plan,
)
from fedot_ind.integration.fedot.compatibility import ensure_fedot_tensor_data


def test_unknown_profile_is_a_structured_error():
    with pytest.raises(IntegrationContractError) as error:
        build_data_plan(
            np.ones((2, 2)),
            profile="dense",
            task=IntegrationTask.CLASSIFICATION,
            stage=DataStage.TRAIN,
        )

    assert error.value.code is IntegrationErrorCode.UNKNOWN_PROFILE
    assert error.value.to_dict()["context"]["allowed"] == ["tensor"]


@pytest.mark.parametrize(
    "profile, data, axes",
    [
        (DataProfile.TENSOR, np.ones((2, 3)), AxisLayout(sample=0, feature=0)),
        (DataProfile.TENSOR, np.ones((2, 3, 4)), AxisLayout(sample=0, feature=1)),
        (DataProfile.TENSOR, np.ones((2, 3)), AxisLayout(sample=2, feature=1)),
    ],
)
def test_invalid_axis_layout_is_rejected(profile, data, axes):
    with pytest.raises(IntegrationContractError) as error:
        build_data_plan(
            data,
            profile=profile,
            task=IntegrationTask.REGRESSION,
            stage=DataStage.TRAIN,
            axes=axes,
        )

    assert error.value.code is IntegrationErrorCode.INVALID_AXES


@pytest.mark.parametrize(
    "profile, data, expected_shape",
    [
        (DataProfile.TENSOR, np.array([[7.0, 8.0]]), (1, 2)),
        (DataProfile.TENSOR, np.array([7.0]), (1, 1)),
    ],
)
def test_single_sample_or_feature_keeps_sample_axis(profile, data, expected_shape):
    plan = build_data_plan(
        data,
        profile=profile,
        task=IntegrationTask.CLASSIFICATION,
        stage=DataStage.TRAIN,
    )

    prepared = normalize_input_data(data, plan)

    assert prepared.values.shape == expected_shape
    assert prepared.idx.tolist() == [0]


def test_pandas_index_and_column_schema_are_preserved():
    frame = pd.DataFrame(
        {"temperature": [1.0, 2.0], "pressure": [3.0, 4.0]},
        index=pd.Index(["row-a", "row-b"], name="sample"),
    )
    plan = build_data_plan(
        frame,
        profile=DataProfile.TENSOR,
        task=IntegrationTask.REGRESSION,
        stage=DataStage.TRAIN,
    )

    prepared = normalize_input_data(frame, plan)

    assert prepared.idx.tolist() == ["row-a", "row-b"]
    assert prepared.schema.columns == ("temperature", "pressure")
    np.testing.assert_array_equal(prepared.values, frame.to_numpy())


def test_predict_columns_must_match_train_columns_and_order():
    train = pd.DataFrame({"left": [1.0, 2.0], "right": [3.0, 4.0]})
    predict = train[["right", "left"]]
    train_plan = build_data_plan(
        train,
        profile=DataProfile.TENSOR,
        task=IntegrationTask.CLASSIFICATION,
        stage=DataStage.TRAIN,
    )
    predict_plan = build_data_plan(
        predict,
        profile=DataProfile.TENSOR,
        task=IntegrationTask.CLASSIFICATION,
        stage=DataStage.PREDICT,
    )

    with pytest.raises(IntegrationContractError) as error:
        validate_prediction_plan(train_plan, predict_plan)

    assert error.value.code is IntegrationErrorCode.SCHEMA_MISMATCH
    assert error.value.to_dict()["context"]["train_schema"]["columns"] == ["left", "right"]


def test_predict_schema_accepts_a_different_number_of_samples():
    train = pd.DataFrame({"left": [1.0, 2.0], "right": [3.0, 4.0]})
    predict = pd.DataFrame({"left": [5.0], "right": [6.0]}, index=["future"])
    train_plan = build_data_plan(
        train,
        profile=DataProfile.TENSOR,
        task=IntegrationTask.CLASSIFICATION,
        stage=DataStage.TRAIN,
    )
    predict_plan = build_data_plan(
        predict,
        profile=DataProfile.TENSOR,
        task=IntegrationTask.CLASSIFICATION,
        stage=DataStage.PREDICT,
    )

    validate_prediction_plan(train_plan, predict_plan)
    prepared = normalize_input_data(predict, predict_plan)

    assert prepared.values.shape == (1, 2)
    assert prepared.idx.tolist() == ["future"]


def test_pandas_axes_cannot_detach_values_from_the_row_index():
    frame = pd.DataFrame({"left": [1.0, 2.0], "right": [3.0, 4.0]})

    with pytest.raises(IntegrationContractError) as error:
        build_data_plan(
            frame,
            profile=DataProfile.TENSOR,
            task=IntegrationTask.REGRESSION,
            stage=DataStage.TRAIN,
            axes=AxisLayout(sample=1, feature=0),
        )

    assert error.value.code is IntegrationErrorCode.INVALID_AXES


def test_normalization_does_not_mutate_numpy_or_pandas_sources():
    array = np.arange(12).reshape(2, 2, 3)
    array_before = array.copy()
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    frame_before = frame.copy(deep=True)
    tensor_plan = build_data_plan(
        array,
        profile=DataProfile.TENSOR,
        task=IntegrationTask.CLASSIFICATION,
        stage=DataStage.TRAIN,
        axes=AxisLayout(sample=1, channel=0, feature=2),
    )
    frame_plan = build_data_plan(
        frame,
        profile=DataProfile.TENSOR,
        task=IntegrationTask.REGRESSION,
        stage=DataStage.TRAIN,
    )

    tensor_result = normalize_input_data(array, tensor_plan)
    frame_result = normalize_input_data(frame, frame_plan)

    np.testing.assert_array_equal(array, array_before)
    pd.testing.assert_frame_equal(frame, frame_before)
    assert tensor_result.values.shape == (2, 2, 3)
    assert not tensor_result.values.flags.writeable
    assert not frame_result.values.flags.writeable


def test_plan_is_deterministic_and_json_serializable():
    frame = pd.DataFrame({"first": [1, 2], "second": [3, 4]})
    kwargs = {
        "profile": DataProfile.TENSOR,
        "task": IntegrationTask.CLASSIFICATION,
        "stage": DataStage.TRAIN,
    }

    first = build_data_plan(frame, **kwargs)
    second = build_data_plan(frame.copy(), **kwargs)

    assert first == second
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()


def test_prediction_batch_copies_inputs_and_is_read_only():
    values = np.array([[0.2, 0.8]])
    idx = np.array([42])
    classes = np.array(["negative", "positive"])

    batch = PredictionBatch(values=values, idx=idx, classes=classes)
    values[0, 0] = 1.0
    idx[0] = 0

    assert batch.values.tolist() == [[0.2, 0.8]]
    assert batch.idx.tolist() == [42]
    assert batch.classes.tolist() == ["negative", "positive"]
    assert not batch.values.flags.writeable
    assert json.loads(json.dumps(batch.to_dict())) == batch.to_dict()


def test_legacy_supervised_data_is_converted_to_tensor_data():
    source = InputData(
        idx=np.arange(4),
        features=np.arange(12, dtype=float).reshape(4, 3),
        target=np.array([0, 1, 0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    converted = ensure_fedot_tensor_data(source, fit_stage=True)

    assert isinstance(converted, TensorData)
    assert tuple(converted.features.shape) == (4, 3)
    assert tuple(converted.target.shape) == (4, 1)


def test_legacy_forecast_series_is_split_by_declared_horizon():
    series = np.arange(12, dtype=float)
    source = InputData(
        idx=np.arange(len(series)),
        features=series,
        target=series,
        task=Task(
            TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length=3),
        ),
        data_type=DataTypesEnum.ts,
    )

    converted = ensure_fedot_tensor_data(source, fit_stage=True)

    assert tuple(converted.features.shape) == (1, 9)
    assert tuple(converted.target.shape) == (1, 3)


def test_forecast_prediction_conversion_requires_fitted_reference_data():
    series = np.arange(12, dtype=float)
    source = InputData(
        idx=np.arange(len(series)),
        features=series,
        target=None,
        task=Task(
            TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length=3),
        ),
        data_type=DataTypesEnum.ts,
    )

    with pytest.raises(ValueError, match="requires reference_data"):
        ensure_fedot_tensor_data(source, fit_stage=False)


def test_prediction_conversion_reuses_training_trace_and_schema():
    task = Task(TaskTypesEnum.regression)
    train_source = InputData(
        idx=np.arange(4),
        features=np.arange(8, dtype=float).reshape(4, 2),
        target=np.arange(4, dtype=float),
        task=task,
        data_type=DataTypesEnum.table,
    )
    predict_source = InputData(
        idx=np.array([10, 11]),
        features=np.arange(4, dtype=float).reshape(2, 2),
        target=None,
        task=task,
        data_type=DataTypesEnum.table,
    )
    train = ensure_fedot_tensor_data(train_source, fit_stage=True)

    predict = ensure_fedot_tensor_data(
        predict_source,
        fit_stage=False,
        reference_data=train,
    )

    assert predict.trace_uuid == train.trace_uuid
    assert tuple(predict.features.shape) == (2, 2)
    assert predict.target is None

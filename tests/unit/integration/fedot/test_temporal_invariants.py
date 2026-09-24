from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fedot_ind.integration.fedot import (
    AnomalyInterval,
    DataProfile,
    DataStage,
    DetectionDataPlan,
    DetectionExecutionPlan,
    DetectionMode,
    DetectionPrediction,
    ForecastingDataPlan,
    ForecastingExecutionPlan,
    ForecastingPrediction,
    IntegrationContractError,
    IntegrationErrorCode,
    PreparedDetectionData,
    PreparedForecastingData,
    RuntimeState,
    TemporalModalityPlan,
    TemporalMultimodalPlan,
    TemporalOrientation,
    TemporalRuntimeSnapshot,
    TemporalSchema,
    TemporalSignalRole,
    build_temporal_multimodal_plan,
    infer_future_time_index,
    labels_to_intervals,
    normalize_anomaly_intervals,
    prepare_detection_data,
    prepare_forecasting_data,
    validate_forecast_coordinates,
)


def _assert_code(error, code: IntegrationErrorCode) -> None:
    assert error.value.code is code


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_execution_plan_rejects_invalid_forecast_horizon(horizon):
    with pytest.raises(IntegrationContractError) as error:
        ForecastingExecutionPlan(
            profile=DataProfile.TENSOR,
            operation_name="lagged_ridge_forecaster",
            horizon=horizon,
        )

    _assert_code(error, IntegrationErrorCode.INVALID_HORIZON)


def test_execution_plans_reject_legacy_profile_and_empty_operation():
    with pytest.raises(IntegrationContractError) as profile_error:
        ForecastingExecutionPlan(
            profile="legacy",
            operation_name="lagged_ridge_forecaster",
            horizon=2,
        )
    _assert_code(profile_error, IntegrationErrorCode.UNKNOWN_PROFILE)

    with pytest.raises(IntegrationContractError) as name_error:
        DetectionExecutionPlan(
            profile=DataProfile.TENSOR,
            operation_name="  ",
        )
    _assert_code(name_error, IntegrationErrorCode.UNSUPPORTED_OPERATION)


def test_detection_execution_plan_normalizes_mode_and_copies_parameters():
    parameters = {
        "layers": (8, 4),
        "families": {"frequency", "statistical"},
        "weights": np.array([0.2, 0.8]),
    }
    plan = DetectionExecutionPlan(
        profile=DataProfile.TENSOR,
        operation_name="feature_iforest_detector",
        mode="scores",
        parameters=parameters,
    )
    parameters["weights"][0] = 9.0

    assert plan.mode is DetectionMode.SCORES
    assert plan.runtime_parameters()["layers"] == [8, 4]
    assert plan.runtime_parameters()["families"] == {
        "frequency", "statistical"}
    np.testing.assert_allclose(plan.runtime_parameters()[
                               "weights"], [0.2, 0.8])
    json.dumps(plan.to_dict())


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"series_count": 0}, IntegrationErrorCode.INVALID_DATA),
        ({"channel_count": 0}, IntegrationErrorCode.INVALID_DATA),
        ({"history_length": 0}, IntegrationErrorCode.INVALID_DATA),
        ({"channel_names": ("only",)}, IntegrationErrorCode.SCHEMA_MISMATCH),
    ],
)
def test_temporal_schema_rejects_invalid_dimensions_and_names(kwargs, code):
    values = {
        "series_count": 1,
        "channel_count": 2,
        "history_length": 5,
        "orientation": TemporalOrientation.TIME_FIRST,
        "source_shape": (5, 2),
        "channel_names": ("left", "right"),
    }
    values.update(kwargs)

    with pytest.raises(IntegrationContractError) as error:
        TemporalSchema(**values)

    _assert_code(error, code)


def test_prepared_forecast_checks_shape_and_both_coordinate_axes():
    schema = TemporalSchema(
        series_count=2,
        channel_count=1,
        history_length=4,
        orientation=TemporalOrientation.TIME_LAST,
        source_shape=(2, 4),
    )
    plan = ForecastingDataPlan(
        profile=DataProfile.TENSOR,
        stage=DataStage.TRAIN,
        horizon=2,
        schema=schema,
    )

    with pytest.raises(IntegrationContractError) as shape_error:
        PreparedForecastingData(
            values=np.ones((1, 4)),
            sample_idx=np.array(["left", "right"]),
            history_time_idx=np.arange(4),
            plan=plan,
        )
    _assert_code(shape_error, IntegrationErrorCode.SCHEMA_MISMATCH)

    with pytest.raises(IntegrationContractError) as sample_error:
        PreparedForecastingData(
            values=np.ones((2, 4)),
            sample_idx=np.array(["left"]),
            history_time_idx=np.arange(4),
            plan=plan,
        )
    _assert_code(sample_error, IntegrationErrorCode.INDEX_MISMATCH)

    with pytest.raises(IntegrationContractError) as time_error:
        PreparedForecastingData(
            values=np.ones((2, 4)),
            sample_idx=np.array(["left", "right"]),
            history_time_idx=np.arange(3),
            plan=plan,
        )
    _assert_code(time_error, IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH)


@pytest.mark.parametrize(
    "values, future, samples, code",
    [
        (np.ones((1, 1, 2)), [4, 5], [0],
         IntegrationErrorCode.LENGTH_MISMATCH),
        (np.ones(2), [4], [0], IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH),
        (np.ones((2, 2)), [4, 5], [0], IntegrationErrorCode.INDEX_MISMATCH),
    ],
)
def test_forecasting_prediction_validates_rank_time_and_series_axes(
        values, future, samples, code,
):
    with pytest.raises(IntegrationContractError) as error:
        ForecastingPrediction(
            values=values,
            forecast_time_idx=np.asarray(future),
            sample_idx=np.asarray(samples),
            horizon=2,
        )

    _assert_code(error, code)


def test_period_and_numeric_indices_extend_without_changing_coordinate_type():
    periods = pd.period_range("2026-01", periods=3, freq="M")
    period_future = infer_future_time_index(periods, 2)
    numeric_future = infer_future_time_index(np.array([1.0, 1.5, 2.0]), 2)

    assert [str(item) for item in period_future] == ["2026-04", "2026-05"]
    np.testing.assert_allclose(numeric_future, [2.5, 3.0])


@pytest.mark.parametrize("history, future", [([], [1]), ([1], []), ([1, 2], [0, 3])])
def test_forecast_coordinate_validation_returns_typed_errors(history, future):
    with pytest.raises(IntegrationContractError) as error:
        validate_forecast_coordinates(np.asarray(history), np.asarray(future))

    assert error.value.code in {
        IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
        IntegrationErrorCode.FUTURE_LEAKAGE,
    }


def test_multimodal_plan_rejects_observed_future_values_and_index_drift():
    target = prepare_forecasting_data(
        np.arange(5), horizon=2, time_idx=np.arange(5), sample_idx=["asset"]
    )
    observed_with_future = prepare_forecasting_data(
        np.arange(7), horizon=2, time_idx=np.arange(7), sample_idx=["asset"]
    )
    shifted = prepare_forecasting_data(
        np.arange(5), horizon=2, time_idx=np.arange(1, 6), sample_idx=["asset"]
    )

    with pytest.raises(IntegrationContractError) as future_error:
        build_temporal_multimodal_plan(
            {
                "target": (target, TemporalSignalRole.TARGET),
                "sensor": (observed_with_future, TemporalSignalRole.OBSERVED),
            },
            horizon=2,
        )
    _assert_code(future_error, IntegrationErrorCode.FUTURE_LEAKAGE)

    with pytest.raises(IntegrationContractError) as time_error:
        build_temporal_multimodal_plan(
            {
                "target": (target, TemporalSignalRole.TARGET),
                "sensor": (shifted, TemporalSignalRole.OBSERVED),
            },
            horizon=2,
        )
    _assert_code(time_error, IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH)


def test_multimodal_plan_rejects_prepared_horizon_mismatch():
    target = prepare_forecasting_data(
        np.arange(5), horizon=3, time_idx=np.arange(5), sample_idx=["asset"]
    )

    with pytest.raises(IntegrationContractError) as error:
        build_temporal_multimodal_plan(
            {"target": (target, TemporalSignalRole.TARGET)},
            horizon=2,
        )

    _assert_code(error, IntegrationErrorCode.PLAN_MISMATCH)


@pytest.mark.parametrize(
    "modalities",
    [
        {},
        {"sensor": ("not-prepared", TemporalSignalRole.OBSERVED)},
    ],
)
def test_multimodal_builder_rejects_empty_or_malformed_payloads(modalities):
    with pytest.raises(IntegrationContractError) as error:
        build_temporal_multimodal_plan(modalities, horizon=2)

    _assert_code(error, IntegrationErrorCode.INVALID_DATA)


def test_temporal_modality_and_collection_validate_future_ownership():
    schema = TemporalSchema(
        series_count=1,
        channel_count=1,
        history_length=4,
        orientation=TemporalOrientation.TIME_FIRST,
        source_shape=(4,),
    )
    target = TemporalModalityPlan("target", TemporalSignalRole.TARGET, schema)

    with pytest.raises(IntegrationContractError) as role_error:
        TemporalModalityPlan(
            "sensor", TemporalSignalRole.OBSERVED, schema, required_future_steps=1
        )
    _assert_code(role_error, IntegrationErrorCode.FUTURE_LEAKAGE)

    with pytest.raises(IntegrationContractError) as key_error:
        TemporalMultimodalPlan(horizon=2, modalities={"wrong": target})
    _assert_code(key_error, IntegrationErrorCode.MODALITY_MISMATCH)


@pytest.mark.parametrize("start, stop", [(-1, 2), (2, 2), (3, 2)])
def test_anomaly_interval_rejects_invalid_half_open_bounds(start, stop):
    with pytest.raises(IntegrationContractError) as error:
        AnomalyInterval(start, stop)

    _assert_code(error, IntegrationErrorCode.INVALID_INTERVAL)


def test_interval_normalization_sorts_disjoint_values_and_rejects_overflow():
    intervals = normalize_anomaly_intervals([(5, 7), (1, 3)], length=8)
    assert intervals == (AnomalyInterval(1, 3), AnomalyInterval(5, 7))

    with pytest.raises(IntegrationContractError) as error:
        normalize_anomaly_intervals([(7, 9)], length=8)
    _assert_code(error, IntegrationErrorCode.INVALID_INTERVAL)


@pytest.mark.parametrize(
    "labels",
    [np.array([[0, 1]]), np.array([0, 2, 0]), np.array(["ok", "bad"])],
)
def test_label_to_interval_conversion_accepts_only_binary_vector(labels):
    with pytest.raises(IntegrationContractError) as error:
        labels_to_intervals(labels)

    _assert_code(error, IntegrationErrorCode.INVALID_INTERVAL)


def test_detection_point_labels_must_match_signal_even_when_all_are_normal():
    with pytest.raises(IntegrationContractError) as error:
        prepare_detection_data(
            np.arange(5), point_labels=np.zeros(4, dtype=int))

    _assert_code(error, IntegrationErrorCode.LENGTH_MISMATCH)


def test_detection_time_last_orientation_preserves_channel_rows():
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=["temperature", "pressure"],
        columns=pd.date_range("2026-03-01", periods=3, freq="h"),
    )

    prepared = prepare_detection_data(
        frame,
        orientation=TemporalOrientation.TIME_LAST,
    )

    assert prepared.values.shape == (3, 2)
    assert prepared.plan.schema.channel_names == ("temperature", "pressure")
    np.testing.assert_array_equal(prepared.time_idx, frame.columns.to_numpy())


def test_prepared_detection_validates_shape_time_intervals_and_mask():
    schema = TemporalSchema(
        series_count=1,
        channel_count=2,
        history_length=4,
        orientation=TemporalOrientation.TIME_FIRST,
        source_shape=(4, 2),
    )
    plan = DetectionDataPlan(
        profile=DataProfile.TENSOR,
        stage=DataStage.TRAIN,
        mode=DetectionMode.LABELS,
        schema=schema,
    )

    cases = [
        (np.ones((4, 1)), np.arange(4), (), None,
         IntegrationErrorCode.SCHEMA_MISMATCH),
        (np.ones((4, 2)), np.arange(3), (), None,
         IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH),
        (np.ones((4, 2)), np.arange(4), (AnomalyInterval(3, 5),), None,
         IntegrationErrorCode.INVALID_INTERVAL),
        (np.ones((4, 2)), np.arange(4), (), np.zeros(3),
         IntegrationErrorCode.LENGTH_MISMATCH),
    ]
    for values, time_idx, intervals, gap_mask, code in cases:
        with pytest.raises(IntegrationContractError) as error:
            PreparedDetectionData(values, time_idx, intervals, plan, gap_mask)
        _assert_code(error, code)


@pytest.mark.parametrize(
    "mode, values, code",
    [
        (DetectionMode.LABELS, np.ones((4, 1)),
         IntegrationErrorCode.SCHEMA_MISMATCH),
        (DetectionMode.SCORES, np.ones((4, 1)),
         IntegrationErrorCode.SCHEMA_MISMATCH),
        (DetectionMode.PROBABILITIES, np.ones(4),
         IntegrationErrorCode.SCHEMA_MISMATCH),
        (DetectionMode.PROBABILITIES, np.ones((4, 3)),
         IntegrationErrorCode.SCHEMA_MISMATCH),
        (DetectionMode.LABELS, np.ones(4), IntegrationErrorCode.LENGTH_MISMATCH),
    ],
)
def test_detection_prediction_shape_follows_output_mode(mode, values, code):
    time_idx = np.arange(
        3) if code is IntegrationErrorCode.LENGTH_MISMATCH else np.arange(4)
    with pytest.raises(IntegrationContractError) as error:
        DetectionPrediction(values=values, time_idx=time_idx, mode=mode)

    _assert_code(error, code)


def test_detection_prediction_rejects_interval_outside_output():
    with pytest.raises(IntegrationContractError) as error:
        DetectionPrediction(
            values=np.zeros(4),
            time_idx=np.arange(4),
            mode=DetectionMode.LABELS,
            intervals=(AnomalyInterval(3, 5),),
        )

    _assert_code(error, IntegrationErrorCode.INVALID_INTERVAL)


def test_runtime_snapshot_is_stable_serializable_state():
    prepared = prepare_detection_data(np.arange(5), mode="scores")
    snapshot = TemporalRuntimeSnapshot(
        profile=DataProfile.TENSOR,
        state=RuntimeState.FITTED,
        operation_name="feature_iforest_detector",
        plan=prepared.plan,
    )

    payload = snapshot.to_dict()
    assert payload["state"] == "fitted"
    assert payload["plan"]["mode"] == "scores"
    json.dumps(payload)

import json

import numpy as np
import pandas as pd
import pytest

from fedot_ind.integration.fedot import (
    AnomalyInterval,
    DataProfile,
    DetectionExecutionPlan,
    DetectionMode,
    ForecastingExecutionPlan,
    ForecastingPrediction,
    IntegrationContractError,
    IntegrationErrorCode,
    IntervalBoundary,
    TemporalOrientation,
    TemporalSignalRole,
    build_temporal_multimodal_plan,
    infer_future_time_index,
    intervals_to_labels,
    labels_to_intervals,
    normalize_anomaly_intervals,
    prepare_detection_data,
    prepare_forecasting_data,
    validate_forecast_coordinates,
)


def test_time_first_dataframe_preserves_calendar_and_channel_coordinates():
    index = pd.date_range("2026-01-01", periods=6, freq="h")
    frame = pd.DataFrame(
        {"temperature": np.arange(6), "pressure": np.arange(10, 16)},
        index=index,
    )

    prepared = prepare_forecasting_data(frame, horizon=2)

    assert prepared.values.shape == (1, 2, 6)
    assert prepared.plan.schema.channel_names == ("temperature", "pressure")
    np.testing.assert_array_equal(prepared.history_time_idx, index.to_numpy())
    assert prepared.sample_idx.tolist() == [0]


def test_time_last_dataframe_keeps_series_rows_separate_from_time_columns():
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=["north", "south"],
        columns=[10, 20, 30],
    )

    prepared = prepare_forecasting_data(
        frame,
        horizon=2,
        orientation=TemporalOrientation.TIME_LAST,
    )

    assert prepared.values.shape == (2, 3)
    assert prepared.sample_idx.tolist() == ["north", "south"]
    assert prepared.history_time_idx.tolist() == [10, 20, 30]


def test_forecasting_normalization_does_not_mutate_source_values():
    source = np.arange(24, dtype=float).reshape(8, 3)
    before = source.copy()

    prepared = prepare_forecasting_data(source, horizon=3)
    source[:] = -1

    np.testing.assert_array_equal(
        before, np.arange(24, dtype=float).reshape(8, 3))
    assert prepared.values[0, 0, 0] == 0.0
    assert not prepared.values.flags.writeable


def test_single_channel_time_first_matrix_uses_univariate_canonical_shape():
    prepared = prepare_forecasting_data(
        np.arange(8, dtype=float).reshape(-1, 1),
        horizon=2,
    )

    assert prepared.values.shape == (1, 8)
    assert prepared.plan.schema.channel_count == 1


@pytest.mark.parametrize("time_idx", [[0, 2, 1], [0, 1, 1]])
def test_forecasting_rejects_non_monotonic_or_duplicate_time_coordinates(time_idx):
    with pytest.raises(IntegrationContractError) as error:
        prepare_forecasting_data(np.arange(3), horizon=1, time_idx=time_idx)

    assert error.value.code is IntegrationErrorCode.INVALID_TEMPORAL_ORDER


def test_future_datetime_index_round_trips_frequency_and_horizon():
    history = pd.date_range("2026-02-01", periods=4, freq="2h")

    future = infer_future_time_index(history, 3)

    np.testing.assert_array_equal(
        future,
        pd.date_range("2026-02-01 08:00", periods=3, freq="2h").to_numpy(),
    )
    validate_forecast_coordinates(history, future)


def test_irregular_index_requires_explicit_step():
    with pytest.raises(IntegrationContractError) as error:
        infer_future_time_index(np.array([0, 1, 3]), 2)

    assert error.value.code is IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH
    assert infer_future_time_index(
        np.array([0, 1, 3]), 2, step=2).tolist() == [5, 7]


@pytest.mark.parametrize(
    "history, step",
    [
        (pd.date_range("2026-01-01", periods=3, freq="D"), "-1D"),
        (pd.period_range("2026-01", periods=3, freq="M"), "-1M"),
    ],
)
def test_calendar_forecast_rejects_negative_steps(history, step):
    with pytest.raises(IntegrationContractError) as error:
        infer_future_time_index(history, 2, step=step)

    assert error.value.code in {
        IntegrationErrorCode.INVALID_TEMPORAL_ORDER,
        IntegrationErrorCode.FUTURE_LEAKAGE,
    }


def test_forecast_coordinates_cannot_overlap_history():
    with pytest.raises(IntegrationContractError) as error:
        validate_forecast_coordinates(np.array([1, 2, 3]), np.array([3, 4]))

    assert error.value.code is IntegrationErrorCode.FUTURE_LEAKAGE


def test_forecasting_prediction_requires_exact_horizon_and_future_coordinates():
    with pytest.raises(IntegrationContractError) as error:
        ForecastingPrediction(
            values=np.array([1.0, 2.0]),
            forecast_time_idx=np.array([10, 11, 12]),
            sample_idx=np.array(["series"]),
            horizon=3,
        )

    assert error.value.code is IntegrationErrorCode.LENGTH_MISMATCH


def test_temporal_multimodal_plan_is_order_invariant_and_checks_future_availability():
    target = prepare_forecasting_data(
        np.arange(6), horizon=2, time_idx=np.arange(6), sample_idx=["asset"]
    )
    observed = prepare_forecasting_data(
        np.arange(6) * 2, horizon=2, time_idx=np.arange(6), sample_idx=["asset"]
    )
    known = prepare_forecasting_data(
        np.arange(8) * 3, horizon=2, time_idx=np.arange(8), sample_idx=["asset"]
    )
    first = build_temporal_multimodal_plan(
        {
            "target": (target, TemporalSignalRole.TARGET),
            "weather": (known, TemporalSignalRole.KNOWN_FUTURE),
            "sensor": (observed, TemporalSignalRole.OBSERVED),
        },
        horizon=2,
    )
    second = build_temporal_multimodal_plan(
        {
            "sensor": (observed, TemporalSignalRole.OBSERVED),
            "weather": (known, TemporalSignalRole.KNOWN_FUTURE),
            "target": (target, TemporalSignalRole.TARGET),
        },
        horizon=2,
    )

    assert first == second
    assert tuple(first.modalities) == ("sensor", "target", "weather")
    assert first.modalities["weather"].required_future_steps == 2

    short_known = prepare_forecasting_data(
        np.arange(7), horizon=2, time_idx=np.arange(7), sample_idx=["asset"]
    )
    with pytest.raises(IntegrationContractError) as error:
        build_temporal_multimodal_plan(
            {
                "target": (target, TemporalSignalRole.TARGET),
                "weather": (short_known, TemporalSignalRole.KNOWN_FUTURE),
            },
            horizon=2,
        )
    assert error.value.code is IntegrationErrorCode.FUTURE_LEAKAGE


def test_known_future_modalities_must_match_the_inferred_forecast_grid():
    target = prepare_forecasting_data(
        np.arange(4),
        horizon=2,
        time_idx=np.arange(4),
        sample_idx=["asset"],
    )
    shifted_future = prepare_forecasting_data(
        np.arange(6),
        horizon=2,
        time_idx=np.array([0, 1, 2, 3, 100, 101]),
        sample_idx=["asset"],
    )

    with pytest.raises(IntegrationContractError) as error:
        build_temporal_multimodal_plan(
            {
                "target": (target, TemporalSignalRole.TARGET),
                "schedule": (
                    shifted_future,
                    TemporalSignalRole.KNOWN_FUTURE,
                ),
            },
            horizon=2,
        )

    assert error.value.code is IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH


def test_anomaly_interval_round_trip_uses_half_open_boundaries():
    labels = np.array([0, 1, 1, 0, 1, 0])

    intervals = labels_to_intervals(labels)
    restored = intervals_to_labels(intervals, len(labels))

    assert intervals == (
        AnomalyInterval(start=1, stop=3),
        AnomalyInterval(start=4, stop=5),
    )
    np.testing.assert_array_equal(restored, labels)


def test_closed_external_intervals_are_normalized_and_overlap_is_rejected():
    assert normalize_anomaly_intervals(
        [(1, 2, "fault")],
        length=5,
        boundary=IntervalBoundary.CLOSED,
    ) == (AnomalyInterval(start=1, stop=3, label="fault"),)

    with pytest.raises(IntegrationContractError) as error:
        normalize_anomaly_intervals([(0, 3), (2, 4)], length=5)
    assert error.value.code is IntegrationErrorCode.INVALID_INTERVAL


def test_detection_data_preserves_datetime_index_and_interval_context():
    index = pd.date_range("2026-03-01", periods=5, freq="min")
    frame = pd.DataFrame({"sensor": [0.0, 0.1, 5.0, 4.0, 0.0]}, index=index)

    prepared = prepare_detection_data(
        frame,
        intervals=[(2, 4)],
        gap_mask=[False, False, False, True, False],
    )

    assert prepared.values.shape == (5, 1)
    np.testing.assert_array_equal(prepared.time_idx, index.to_numpy())
    assert prepared.intervals == (AnomalyInterval(2, 4),)
    assert prepared.plan.schema.channel_names == ("sensor",)


def test_detection_rejects_conflicting_point_and_interval_labels():
    with pytest.raises(IntegrationContractError) as error:
        prepare_detection_data(
            np.arange(5),
            intervals=[(1, 3)],
            point_labels=[0, 0, 1, 0, 0],
        )

    assert error.value.code is IntegrationErrorCode.INVALID_INTERVAL


def test_temporal_execution_plans_own_immutable_parameter_snapshots():
    source = {"window_size": 12, "nested": {"alpha": [1.0]}}
    forecast = ForecastingExecutionPlan(
        profile=DataProfile.TENSOR,
        operation_name=" lagged_ridge_forecaster ",
        horizon=3,
        parameters=source,
    )
    detection = DetectionExecutionPlan(
        profile=DataProfile.TENSOR,
        operation_name="feature_iforest_detector",
        mode=DetectionMode.SCORES,
        parameters=source,
    )
    source["nested"]["alpha"].append(2.0)

    assert forecast.operation_name == "lagged_ridge_forecaster"
    assert forecast.parameters["nested"]["alpha"] == (1.0,)
    assert detection.parameters["nested"]["alpha"] == (1.0,)
    assert forecast.runtime_parameters()["nested"]["alpha"] == [1.0]
    assert json.loads(json.dumps(forecast.to_dict())) == forecast.to_dict()


def test_tensor_data_keeps_sample_and_temporal_indices_in_separate_layers():
    import fedot

    if not hasattr(fedot, "create_data"):
        pytest.skip("The FEDOT TensorData profile is not installed.")
    from fedot_ind.integration.fedot import (
        create_detection_tensor_data,
        create_forecasting_tensor_data,
    )

    forecast = prepare_forecasting_data(
        pd.Series(
            np.arange(8, dtype=float),
            index=pd.date_range("2026-04-01", periods=8, freq="D"),
        ),
        horizon=2,
        sample_idx=["series-a"],
    )
    detection = prepare_detection_data(
        pd.DataFrame(
            {"sensor": np.arange(8, dtype=float)},
            index=pd.date_range("2026-04-01", periods=8, freq="D"),
        ),
    )

    forecast_tensor = create_forecasting_tensor_data(forecast)
    detection_tensor = create_detection_tensor_data(detection)

    assert np.asarray(forecast_tensor.tensor_data.idx).tolist() == ["series-a"]
    assert len(forecast_tensor.prepared.history_time_idx) == 8
    assert len(np.asarray(detection_tensor.tensor_data.idx)) == 8

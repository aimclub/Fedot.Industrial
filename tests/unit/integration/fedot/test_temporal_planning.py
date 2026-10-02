"""Boundary cases for temporal coordinates and anomaly annotations."""

import numpy as np
import pandas as pd
import pytest

from fedot_ind.integration.fedot import (
    AnomalyInterval,
    IntegrationContractError,
    IntegrationErrorCode,
    infer_future_time_index,
    intervals_to_labels,
    labels_to_intervals,
    normalize_anomaly_intervals,
    prepare_detection_data,
    prepare_forecasting_data,
)


@pytest.mark.parametrize("horizon", [False, 0, -1, 1.5, "2", None])
@pytest.mark.parametrize("prepare", [infer_future_time_index, prepare_forecasting_data])
def test_temporal_planning_rejects_non_positive_integer_horizons(prepare, horizon):
    with pytest.raises(IntegrationContractError) as error:
        prepare(np.arange(4), horizon=horizon)

    assert error.value.code is IntegrationErrorCode.INVALID_HORIZON


def test_single_history_coordinate_requires_step_and_accepts_numpy_horizon():
    with pytest.raises(IntegrationContractError) as error:
        infer_future_time_index([10], 1)

    assert error.value.code is IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH
    np.testing.assert_array_equal(infer_future_time_index([10], np.int64(2), step=3), [13, 16])


def test_month_end_forecast_obeys_calendar_instead_of_fixed_day_deltas():
    history = pd.date_range("2024-01-31", periods=3, freq=pd.offsets.MonthEnd())

    future = infer_future_time_index(history, 2)

    np.testing.assert_array_equal(future, pd.to_datetime(["2024-04-30", "2024-05-31"]).to_numpy())


def test_timezone_aware_forecast_crosses_daylight_saving_without_duplicate_times():
    history = pd.date_range("2024-03-10 00:00", periods=2, freq="h", tz="America/New_York")

    future = infer_future_time_index(history, 3)

    expected = pd.date_range("2024-03-10 03:00", periods=3, freq="h", tz="America/New_York")
    np.testing.assert_array_equal(future, expected.to_numpy())


@pytest.mark.parametrize("step", [0, -1])
def test_numeric_step_cannot_move_forecast_into_history(step):
    with pytest.raises(IntegrationContractError) as error:
        infer_future_time_index([1, 2, 3], 1, step=step)

    assert error.value.code is IntegrationErrorCode.FUTURE_LEAKAGE


@pytest.mark.parametrize("orientation", ["time_first", "time_last"])
def test_batched_multichannel_forecast_keeps_series_and_time_separate(orientation):
    canonical = np.arange(30).reshape(2, 3, 5)
    source = canonical.transpose(2, 0, 1).copy() if orientation == "time_first" else canonical.copy()
    prepared = prepare_forecasting_data(
        source, horizon=1, orientation=orientation,
        sample_idx=["north", "south"], time_idx=[10, 20, 30, 40, 50],
        channel_names=["a", "b", "c"],
    )
    source[...] = -1

    np.testing.assert_array_equal(prepared.values, canonical)
    np.testing.assert_array_equal(prepared.sample_idx, ["north", "south"])
    np.testing.assert_array_equal(prepared.history_time_idx, [10, 20, 30, 40, 50])
    assert prepared.plan.schema.channel_names == ("a", "b", "c")


@pytest.mark.parametrize("labels, bounds", [
    ([], []),
    ([0], []),
    ([1], [(0, 1)]),
    ([1, 1, 1], [(0, 3)]),
    ([1, 0, 1], [(0, 1), (2, 3)]),
    ([False, True, True, False], [(1, 3)]),
])
def test_binary_interval_round_trip_preserves_signal_boundaries(labels, bounds):
    intervals = labels_to_intervals(labels)

    assert intervals == tuple(AnomalyInterval(start, stop) for start, stop in bounds)
    restored = intervals_to_labels(intervals, len(labels))
    np.testing.assert_array_equal(restored, labels)
    assert not restored.flags.writeable


def test_adjacent_intervals_are_allowed_and_expand_without_a_gap():
    intervals = normalize_anomaly_intervals([(2, 4, "second"), (0, 2, "first")], length=4)

    assert intervals == (AnomalyInterval(0, 2, "first"), AnomalyInterval(2, 4, "second"))
    np.testing.assert_array_equal(intervals_to_labels(intervals, 4), [1, 1, 1, 1])


def test_closed_singleton_at_last_point_is_converted_exactly_once():
    canonical = AnomalyInterval(2, 3)

    assert normalize_anomaly_intervals([(2, 2)], length=3, boundary="closed") == (canonical,)
    assert normalize_anomaly_intervals([canonical], length=3, boundary="closed") == (canonical,)


def test_prepared_detection_owns_values_time_and_gap_mask():
    values = np.arange(4, dtype=float)
    time_idx = np.arange(10, 14)
    gap_mask = np.array([False, True, False, False])
    prepared = prepare_detection_data(values, time_idx=time_idx, gap_mask=gap_mask)
    values[:] = -1
    time_idx[:] = -1
    gap_mask[:] = True

    np.testing.assert_array_equal(prepared.values[:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(prepared.time_idx, [10, 11, 12, 13])
    np.testing.assert_array_equal(prepared.gap_mask, [False, True, False, False])
    for snapshot in (prepared.values, prepared.time_idx, prepared.gap_mask):
        assert not snapshot.flags.writeable

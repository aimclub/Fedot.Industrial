"""Pure planning and normalization for temporal FEDOT boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fedot_ind.integration.fedot.contracts import (
    DataProfile,
    DataStage,
    IntegrationContractError,
    IntegrationErrorCode,
)
from fedot_ind.integration.fedot.temporal_contracts import (
    AnomalyInterval,
    DetectionDataPlan,
    DetectionMode,
    ForecastingDataPlan,
    IntervalBoundary,
    PreparedDetectionData,
    PreparedForecastingData,
    TemporalModalityPlan,
    TemporalMultimodalPlan,
    TemporalOrientation,
    TemporalSchema,
    TemporalSignalRole,
)


TemporalSource = np.ndarray | pd.Series | pd.DataFrame | Sequence[float]


def prepare_forecasting_data(
        data: TemporalSource,
        *,
        horizon: int,
        stage: DataStage | str = DataStage.TRAIN,
        orientation: TemporalOrientation | str = TemporalOrientation.TIME_FIRST,
        time_idx: Sequence[Any] | np.ndarray | pd.Index | None = None,
        sample_idx: Sequence[Any] | np.ndarray | pd.Index | None = None,
        channel_names: Sequence[str] | None = None,
) -> PreparedForecastingData:
    """Normalize a temporal source without conflating object and time indices."""
    normalized_stage = _parse_enum(
        stage, DataStage, IntegrationErrorCode.UNKNOWN_STAGE, "stage")
    normalized_orientation = _parse_enum(
        orientation,
        TemporalOrientation,
        IntegrationErrorCode.INVALID_AXES,
        "orientation",
    )
    values, inferred_time, inferred_samples, inferred_channels = _canonical_forecasting_values(
        data,
        normalized_orientation,
    )
    resolved_time = _resolve_index(
        time_idx,
        inferred_time,
        values.shape[-1],
        "history_time_idx",
    )
    resolved_samples = _resolve_index(
        sample_idx,
        inferred_samples,
        values.shape[0],
        "sample_idx",
    )
    _validate_temporal_order(resolved_time, "history_time_idx")
    resolved_channels = _resolve_channel_names(
        channel_names,
        inferred_channels,
        1 if values.ndim == 2 else values.shape[1],
    )
    schema = TemporalSchema(
        series_count=values.shape[0],
        channel_count=1 if values.ndim == 2 else values.shape[1],
        history_length=values.shape[-1],
        orientation=normalized_orientation,
        source_shape=tuple(int(size) for size in np.asarray(data).shape),
        channel_names=resolved_channels,
    )
    plan = ForecastingDataPlan(
        profile=DataProfile.TENSOR,
        stage=normalized_stage,
        horizon=_validate_horizon(horizon),
        schema=schema,
    )
    return PreparedForecastingData(
        values=values,
        sample_idx=resolved_samples,
        history_time_idx=resolved_time,
        plan=plan,
    )


def infer_future_time_index(
        history_time_idx: Sequence[Any] | np.ndarray | pd.Index,
        horizon: int,
        *,
        step: Any | None = None,
) -> np.ndarray:
    """Extend a regular temporal index by exactly ``horizon`` coordinates."""
    resolved_horizon = _validate_horizon(horizon)
    history = np.asarray(history_time_idx)
    _validate_temporal_order(history, "history_time_idx")
    if history.size < 1:
        raise IntegrationContractError(
            IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
            "A forecast requires at least one history coordinate.",
        )

    pandas_index = pd.Index(history_time_idx)
    if isinstance(pandas_index, pd.PeriodIndex):
        frequency = step or pandas_index.freq
        if frequency is None:
            raise _frequency_error(history)
        frequency = _positive_calendar_offset(frequency, history)
        start = pandas_index[-1] + 1
        future = pd.period_range(
            start=start,
            periods=resolved_horizon,
            freq=frequency,
        ).to_numpy()
        validate_forecast_coordinates(history, future)
        return future
    if isinstance(pandas_index, pd.DatetimeIndex):
        frequency = step or pandas_index.freq or pd.infer_freq(pandas_index)
        if frequency is None:
            frequency = _constant_step(history)
        if frequency is None:
            raise _frequency_error(history)
        offset = _positive_calendar_offset(frequency, history)
        future = pd.date_range(
            start=pandas_index[-1] + offset,
            periods=resolved_horizon,
            freq=offset,
        ).to_numpy()
        validate_forecast_coordinates(history, future)
        return future

    resolved_step = step if step is not None else _constant_step(history)
    if resolved_step is None:
        raise _frequency_error(history)
    try:
        future = np.asarray([history[-1] + resolved_step * number
                             for number in range(1, resolved_horizon + 1)])
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
            "The supplied temporal step cannot extend the history index.",
            context={"step": str(resolved_step), "dtype": str(history.dtype)},
            cause=error,
        ) from error
    validate_forecast_coordinates(history, future)
    return future


def validate_forecast_coordinates(
        history_time_idx: Sequence[Any] | np.ndarray,
        forecast_time_idx: Sequence[Any] | np.ndarray,
) -> None:
    """Reject coordinates that overlap history or move backwards in time."""
    history = np.asarray(history_time_idx)
    future = np.asarray(forecast_time_idx)
    _validate_temporal_order(history, "history_time_idx")
    _validate_temporal_order(future, "forecast_time_idx")
    if history.size == 0 or future.size == 0:
        raise IntegrationContractError(
            IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
            "History and forecast coordinates must both be non-empty.",
        )
    try:
        overlaps = bool(np.intersect1d(history, future).size)
        follows_history = bool(future[0] > history[-1])
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
            "History and forecast coordinates are not comparable.",
            cause=error,
        ) from error
    if overlaps or not follows_history:
        raise IntegrationContractError(
            IntegrationErrorCode.FUTURE_LEAKAGE,
            "Forecast coordinates must begin strictly after the history.",
            context={"history_end": str(
                history[-1]), "forecast_start": str(future[0])},
        )


def build_temporal_multimodal_plan(
        modalities: Mapping[str, tuple[PreparedForecastingData, TemporalSignalRole | str]],
        *,
        horizon: int,
) -> TemporalMultimodalPlan:
    """Validate aligned target, observed, and known-future temporal signals."""
    resolved_horizon = _validate_horizon(horizon)
    if not isinstance(modalities, Mapping) or not modalities:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            "Temporal multimodal input must be a non-empty mapping.",
        )
    normalized: dict[str,
                     tuple[PreparedForecastingData, TemporalSignalRole]] = {}
    for raw_name, payload in modalities.items():
        if not isinstance(raw_name, str) or not raw_name.strip() or raw_name != raw_name.strip():
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Temporal modality names must be trimmed non-empty strings.",
                context={"modality": raw_name},
            )
        if not isinstance(payload, tuple) or len(payload) != 2 or not isinstance(payload[0], PreparedForecastingData):
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Each temporal modality must pair prepared data with a signal role.",
                context={"modality": raw_name},
            )
        role = _parse_enum(
            payload[1],
            TemporalSignalRole,
            IntegrationErrorCode.MODALITY_MISMATCH,
            "signal_role",
        )
        normalized[raw_name] = payload[0], role

    targets = [(name, data) for name, (data, role) in normalized.items()
               if role is TemporalSignalRole.TARGET]
    if len(targets) != 1:
        raise IntegrationContractError(
            IntegrationErrorCode.MODALITY_MISMATCH,
            "Temporal multimodal input requires exactly one target signal.",
            context={"targets": [name for name, _ in targets]},
        )
    _, target = targets[0]
    target_length = target.plan.schema.history_length
    expected_future = _expected_multimodal_future_grid(
        target.history_time_idx,
        resolved_horizon,
    )
    declarations: dict[str, TemporalModalityPlan] = {}
    for name, (data, role) in normalized.items():
        if data.plan.horizon != resolved_horizon:
            raise IntegrationContractError(
                IntegrationErrorCode.PLAN_MISMATCH,
                "Every temporal modality must use the requested forecast horizon.",
                context={
                    "modality": name,
                    "requested": resolved_horizon,
                    "prepared": data.plan.horizon,
                },
            )
        if not np.array_equal(data.sample_idx, target.sample_idx):
            raise IntegrationContractError(
                IntegrationErrorCode.INDEX_MISMATCH,
                "All temporal modalities must use the same object coordinates.",
                context={"modality": name},
            )
        available_future = max(
            0, data.plan.schema.history_length - target_length)
        history_prefix = data.history_time_idx[:target_length]
        if data.plan.schema.history_length < target_length or not np.array_equal(
                history_prefix, target.history_time_idx):
            raise IntegrationContractError(
                IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
                "Temporal modalities must share the target history grid.",
                context={"modality": name},
            )
        if role is not TemporalSignalRole.KNOWN_FUTURE and available_future:
            raise IntegrationContractError(
                IntegrationErrorCode.FUTURE_LEAKAGE,
                "Observed signals cannot expose forecast-period values.",
                context={"modality": name, "future_steps": available_future},
            )
        if role is TemporalSignalRole.KNOWN_FUTURE:
            if available_future < resolved_horizon:
                raise IntegrationContractError(
                    IntegrationErrorCode.FUTURE_LEAKAGE,
                    "Known-future signals must cover the complete forecast horizon.",
                    context={"modality": name, "required": resolved_horizon,
                             "available": available_future},
                )
            future_grid = data.history_time_idx[
                target_length:target_length + resolved_horizon
            ]
            validate_forecast_coordinates(target.history_time_idx, future_grid)
            if expected_future is None:
                expected_future = np.array(future_grid, copy=True)
            elif not np.array_equal(future_grid, expected_future):
                raise IntegrationContractError(
                    IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
                    "Known-future modalities must share one forecast grid.",
                    context={
                        "modality": name,
                        "expected": expected_future.tolist(),
                        "actual": np.asarray(future_grid).tolist(),
                    },
                )
        declarations[name] = TemporalModalityPlan(
            name=name,
            role=role,
            schema=data.plan.schema,
            required_future_steps=available_future if role is TemporalSignalRole.KNOWN_FUTURE else 0,
        )
    return TemporalMultimodalPlan(horizon=resolved_horizon, modalities=declarations)


def _expected_multimodal_future_grid(
        history_time_idx: np.ndarray,
        horizon: int,
) -> np.ndarray | None:
    """Infer the shared future grid when the target history is regular."""
    try:
        return infer_future_time_index(history_time_idx, horizon)
    except IntegrationContractError as error:
        if error.code is IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH:
            return None
        raise


def prepare_detection_data(
        data: TemporalSource,
        *,
        stage: DataStage | str = DataStage.TRAIN,
        mode: DetectionMode | str = DetectionMode.LABELS,
        orientation: TemporalOrientation | str = TemporalOrientation.TIME_FIRST,
        time_idx: Sequence[Any] | np.ndarray | pd.Index | None = None,
        intervals: Sequence[AnomalyInterval | Sequence[Any]] = (),
        interval_boundary: IntervalBoundary | str = IntervalBoundary.HALF_OPEN,
        point_labels: Sequence[int] | np.ndarray | None = None,
        gap_mask: Sequence[bool] | np.ndarray | None = None,
        channel_names: Sequence[str] | None = None,
        causal: bool = True,
) -> PreparedDetectionData:
    """Normalize detector input and interval labels on one explicit time grid."""
    normalized_stage = _parse_enum(
        stage, DataStage, IntegrationErrorCode.UNKNOWN_STAGE, "stage")
    normalized_mode = _parse_enum(
        mode, DetectionMode, IntegrationErrorCode.UNSUPPORTED_OUTPUT_MODE, "mode")
    normalized_orientation = _parse_enum(
        orientation,
        TemporalOrientation,
        IntegrationErrorCode.INVALID_AXES,
        "orientation",
    )
    values, inferred_time, inferred_channels = _canonical_detection_values(
        data, normalized_orientation)
    resolved_time = _resolve_index(
        time_idx, inferred_time, values.shape[0], "time_idx")
    _validate_temporal_order(resolved_time, "time_idx")
    resolved_channels = _resolve_channel_names(
        channel_names,
        inferred_channels,
        values.shape[1],
    )
    schema = TemporalSchema(
        series_count=1,
        channel_count=values.shape[1],
        history_length=values.shape[0],
        orientation=normalized_orientation,
        source_shape=tuple(int(size) for size in np.asarray(data).shape),
        channel_names=resolved_channels,
    )
    plan = DetectionDataPlan(
        profile=DataProfile.TENSOR,
        stage=normalized_stage,
        mode=normalized_mode,
        schema=schema,
        causal=bool(causal),
    )
    normalized_intervals = normalize_anomaly_intervals(
        intervals,
        length=values.shape[0],
        boundary=interval_boundary,
    )
    if point_labels is not None:
        label_array = np.asarray(point_labels)
        if label_array.ndim != 1 or len(label_array) != values.shape[0]:
            raise IntegrationContractError(
                IntegrationErrorCode.LENGTH_MISMATCH,
                "Point labels must match the detection point count.",
                context={"labels": list(label_array.shape),
                         "points": values.shape[0]},
            )
        label_intervals = labels_to_intervals(label_array)
        if normalized_intervals and normalized_intervals != label_intervals:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_INTERVAL,
                "Point labels and interval labels describe different anomalies.",
            )
        normalized_intervals = label_intervals
    return PreparedDetectionData(
        values=values,
        time_idx=resolved_time,
        intervals=normalized_intervals,
        plan=plan,
        gap_mask=None if gap_mask is None else np.asarray(
            gap_mask, dtype=bool),
    )


def normalize_anomaly_intervals(
        intervals: Sequence[AnomalyInterval | Sequence[Any]],
        *,
        length: int,
        boundary: IntervalBoundary | str = IntervalBoundary.HALF_OPEN,
) -> tuple[AnomalyInterval, ...]:
    """Convert external interval conventions to sorted half-open intervals."""
    normalized_boundary = _parse_enum(
        boundary,
        IntervalBoundary,
        IntegrationErrorCode.INVALID_INTERVAL,
        "interval_boundary",
    )
    normalized: list[AnomalyInterval] = []
    for value in intervals:
        if isinstance(value, AnomalyInterval):
            interval = value
        else:
            parts = tuple(value)
            if len(parts) not in (2, 3):
                raise IntegrationContractError(
                    IntegrationErrorCode.INVALID_INTERVAL,
                    "Anomaly interval must contain start, stop, and an optional label.",
                    context={"interval": list(parts)},
                )
            start, stop = int(parts[0]), int(parts[1])
            if normalized_boundary is IntervalBoundary.CLOSED:
                stop += 1
            interval = AnomalyInterval(start=start, stop=stop,
                                       label="anomaly" if len(parts) == 2 else str(parts[2]))
        if interval.stop > length:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_INTERVAL,
                "Anomaly interval exceeds the signal length.",
                context={"interval": interval.to_dict(), "length": length},
            )
        normalized.append(interval)
    normalized.sort(key=lambda item: (item.start, item.stop, item.label))
    for previous, current in zip(normalized, normalized[1:]):
        if current.start < previous.stop:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_INTERVAL,
                "Anomaly intervals must not overlap.",
                context={"previous": previous.to_dict(
                ), "current": current.to_dict()},
            )
    return tuple(normalized)


def labels_to_intervals(labels: Sequence[int] | np.ndarray) -> tuple[AnomalyInterval, ...]:
    """Convert binary point labels to canonical half-open intervals."""
    values = np.asarray(labels)
    if values.ndim != 1 or not set(np.unique(values)).issubset({0, 1, False, True}):
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_INTERVAL,
            "Point anomaly labels must be a one-dimensional binary vector.",
            context={"shape": list(values.shape)},
        )
    result: list[AnomalyInterval] = []
    start: int | None = None
    for position, value in enumerate(values.astype(bool)):
        if value and start is None:
            start = position
        elif not value and start is not None:
            result.append(AnomalyInterval(start=start, stop=position))
            start = None
    if start is not None:
        result.append(AnomalyInterval(start=start, stop=len(values)))
    return tuple(result)


def intervals_to_labels(intervals: Sequence[AnomalyInterval], length: int) -> np.ndarray:
    """Expand canonical intervals into a read-only binary point vector."""
    normalized = normalize_anomaly_intervals(intervals, length=length)
    labels = np.zeros(length, dtype=int)
    for interval in normalized:
        labels[interval.start:interval.stop] = 1
    labels.setflags(write=False)
    return labels


def _canonical_forecasting_values(
        data: TemporalSource,
        orientation: TemporalOrientation,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, tuple[str, ...]]:
    if isinstance(data, pd.Series):
        if orientation is not TemporalOrientation.TIME_FIRST:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_AXES,
                "A pandas Series uses rows as its temporal axis.",
            )
        return data.to_numpy(copy=True).reshape(1, -1), data.index.to_numpy(copy=True), None, ()
    if isinstance(data, pd.DataFrame):
        raw = data.to_numpy(copy=True)
        if orientation is TemporalOrientation.TIME_FIRST:
            canonical = raw.T.reshape(1, raw.shape[1], raw.shape[0])
            if canonical.shape[1] == 1:
                canonical = canonical.reshape(1, canonical.shape[-1])
            return canonical, data.index.to_numpy(copy=True), None, tuple(str(item) for item in data.columns)
        return raw, data.columns.to_numpy(copy=True), data.index.to_numpy(copy=True), ()

    raw = np.asarray(data)
    if raw.ndim == 1:
        return np.array(raw, copy=True).reshape(1, -1), None, None, ()
    if raw.ndim == 2:
        if orientation is TemporalOrientation.TIME_FIRST:
            canonical = np.transpose(raw, (1, 0)).reshape(
                1, raw.shape[1], raw.shape[0]
            )
            if canonical.shape[1] == 1:
                canonical = canonical.reshape(1, canonical.shape[-1])
            return canonical, None, None, ()
        return np.array(raw, copy=True), None, None, ()
    if raw.ndim == 3:
        canonical = np.transpose(
            raw, (1, 2, 0)) if orientation is TemporalOrientation.TIME_FIRST else raw
        return np.array(canonical, copy=True), None, None, ()
    raise IntegrationContractError(
        IntegrationErrorCode.INVALID_DATA,
        "Forecasting data must have one, two, or three dimensions.",
        context={"rank": raw.ndim},
    )


def _canonical_detection_values(
        data: TemporalSource,
        orientation: TemporalOrientation,
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...]]:
    if isinstance(data, pd.Series):
        if orientation is not TemporalOrientation.TIME_FIRST:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_AXES,
                "A pandas Series uses rows as its temporal axis.",
            )
        return data.to_numpy(copy=True).reshape(-1, 1), data.index.to_numpy(copy=True), ()
    if isinstance(data, pd.DataFrame):
        values = data.to_numpy(copy=True)
        if orientation is TemporalOrientation.TIME_LAST:
            values = values.T
            time_values = data.columns.to_numpy(copy=True)
            channel_values = tuple(str(item) for item in data.index)
        else:
            time_values = data.index.to_numpy(copy=True)
            channel_values = tuple(str(item) for item in data.columns)
        return values, time_values, channel_values
    values = np.asarray(data)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim == 2 and orientation is TemporalOrientation.TIME_LAST:
        values = values.T
    elif values.ndim != 2:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            "Detection data must have one or two dimensions.",
            context={"rank": values.ndim},
        )
    return np.array(values, copy=True), None, ()


def _resolve_index(
        explicit: Sequence[Any] | np.ndarray | pd.Index | None,
        inferred: np.ndarray | None,
        expected_length: int,
        field: str,
) -> np.ndarray:
    values = np.asarray(explicit if explicit is not None else inferred
                        if inferred is not None else np.arange(expected_length))
    if values.ndim != 1 or len(values) != expected_length:
        raise IntegrationContractError(
            IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH if "time" in field
            else IntegrationErrorCode.INDEX_MISMATCH,
            f"{field} must be one-dimensional and match its axis.",
            context={"field": field, "shape": list(
                values.shape), "expected": expected_length},
        )
    return np.array(values, copy=True)


def _resolve_channel_names(
        explicit: Sequence[str] | None,
        inferred: tuple[str, ...],
        channel_count: int,
) -> tuple[str, ...]:
    names = tuple(str(item)
                  for item in explicit) if explicit is not None else inferred
    if names and (len(names) != channel_count or len(set(names)) != len(names)):
        raise IntegrationContractError(
            IntegrationErrorCode.SCHEMA_MISMATCH,
            "Channel names must be unique and match the channel count.",
            context={"channel_count": channel_count,
                     "channel_names": list(names)},
        )
    return names


def _validate_temporal_order(values: np.ndarray, field: str) -> None:
    index = pd.Index(values)
    if not index.is_unique or not index.is_monotonic_increasing:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_TEMPORAL_ORDER,
            "Temporal coordinates must be unique and monotonically increasing.",
            context={"field": field},
        )


def _constant_step(values: np.ndarray) -> Any | None:
    if len(values) < 2:
        return None
    try:
        differences = np.diff(values)
        if np.all(differences == differences[0]) and differences[0] > 0:
            return differences[0]
    except (TypeError, ValueError):
        return None
    return None


def _frequency_error(values: np.ndarray) -> IntegrationContractError:
    return IntegrationContractError(
        IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
        "Future coordinates require a regular history index or an explicit step.",
        context={"coordinates": [str(item) for item in values.tolist()]},
    )


def _positive_calendar_offset(frequency: Any, history: np.ndarray):
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
            "The supplied calendar step is invalid.",
            context={"step": str(frequency), "dtype": str(history.dtype)},
            cause=error,
        ) from error
    if int(offset.n) <= 0:
        raise IntegrationContractError(
            IntegrationErrorCode.FUTURE_LEAKAGE,
            "A forecast calendar step must move forward in time.",
            context={"step": str(frequency)},
        )
    return offset


def _validate_horizon(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_HORIZON,
            "Forecast horizon must be a positive integer.",
            context={"horizon": value},
        )
    return int(value)


def _parse_enum(value: Any, enum_type, code: IntegrationErrorCode, field: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            code,
            f"Unsupported {field} value.",
            context={"field": field, "value": value,
                     "allowed": [item.value for item in enum_type]},
            cause=error,
        ) from error

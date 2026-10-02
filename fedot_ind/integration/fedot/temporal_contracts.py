"""Typed values for forecasting and anomaly-detection integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fedot_ind.integration.fedot.contracts import (
    DataProfile,
    DataStage,
    IntegrationContractError,
    IntegrationErrorCode,
    RuntimeState,
)
from fedot_ind.integration.fedot.parameter_codec import (
    freeze_mapping as _freeze_mapping,
    thaw_value as _thaw_value,
)


class TemporalOrientation(str, Enum):
    """Location of the temporal dimension in a supplied array."""

    TIME_FIRST = "time_first"
    TIME_LAST = "time_last"


class TemporalSignalRole(str, Enum):
    """Availability contract for one temporal modality."""

    TARGET = "target"
    OBSERVED = "observed"
    KNOWN_FUTURE = "known_future"


class DetectionMode(str, Enum):
    """Supported anomaly-detection result representations."""

    LABELS = "labels"
    SCORES = "scores"
    PROBABILITIES = "probs"


class IntervalBoundary(str, Enum):
    """Convention used by externally supplied anomaly intervals."""

    CLOSED = "closed"
    HALF_OPEN = "half_open"


def _normalize_profile(value: object) -> DataProfile:
    try:
        return value if isinstance(value, DataProfile) else DataProfile(value)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.UNKNOWN_PROFILE,
            "Unsupported FEDOT integration profile.",
            context={"profile": str(value)},
        ) from error


@dataclass(frozen=True)
class ForecastingExecutionPlan:
    """Immutable forecasting model selection and horizon contract."""

    profile: DataProfile
    operation_name: str
    horizon: int
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        profile = _normalize_profile(self.profile)
        object.__setattr__(self, "profile", profile)
        if profile is not DataProfile.TENSOR:
            raise IntegrationContractError(
                IntegrationErrorCode.UNKNOWN_PROFILE,
                "Forecasting execution requires the TensorData profile.",
                context={"profile": profile.value},
            )
        name = self.operation_name.strip()
        if not name:
            raise IntegrationContractError(
                IntegrationErrorCode.UNSUPPORTED_OPERATION,
                "Forecasting operation name must not be empty.",
            )
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_HORIZON,
                "Forecast horizon must be a positive integer.",
                context={"horizon": self.horizon},
            )
        object.__setattr__(self, "operation_name", name)
        object.__setattr__(self, "parameters",
                           _freeze_mapping(self.parameters))

    def runtime_parameters(self) -> dict[str, Any]:
        return {key: _thaw_value(value) for key, value in self.parameters.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "operation_name": self.operation_name,
            "horizon": self.horizon,
            "parameters": _jsonable(self.parameters),
        }


@dataclass(frozen=True)
class DetectionExecutionPlan:
    """Immutable detector selection and point-output contract."""

    profile: DataProfile
    operation_name: str
    mode: DetectionMode = DetectionMode.LABELS
    causal: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        profile = _normalize_profile(self.profile)
        object.__setattr__(self, "profile", profile)
        if profile is not DataProfile.TENSOR:
            raise IntegrationContractError(
                IntegrationErrorCode.UNKNOWN_PROFILE,
                "Detection execution requires the TensorData profile.",
                context={"profile": profile.value},
            )
        name = self.operation_name.strip()
        if not name:
            raise IntegrationContractError(
                IntegrationErrorCode.UNSUPPORTED_OPERATION,
                "Detection operation name must not be empty.",
            )
        if not isinstance(self.mode, DetectionMode):
            try:
                object.__setattr__(self, "mode", DetectionMode(self.mode))
            except (TypeError, ValueError) as error:
                raise IntegrationContractError(
                    IntegrationErrorCode.UNSUPPORTED_OUTPUT_MODE,
                    "Detection output mode is unsupported.",
                    context={"mode": self.mode},
                    cause=error,
                ) from error
        object.__setattr__(self, "operation_name", name)
        object.__setattr__(self, "parameters",
                           _freeze_mapping(self.parameters))

    def runtime_parameters(self) -> dict[str, Any]:
        return {key: _thaw_value(value) for key, value in self.parameters.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "operation_name": self.operation_name,
            "mode": self.mode.value,
            "causal": self.causal,
            "parameters": _jsonable(self.parameters),
        }


@dataclass(frozen=True)
class TemporalSchema:
    """Canonical structure of one temporal signal."""

    series_count: int
    channel_count: int
    history_length: int
    orientation: TemporalOrientation
    source_shape: tuple[int, ...]
    channel_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.series_count < 1 or self.channel_count < 1 or self.history_length < 1:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Temporal dimensions must be positive.",
                context={
                    "series_count": self.series_count,
                    "channel_count": self.channel_count,
                    "history_length": self.history_length,
                },
            )
        if self.channel_names and len(self.channel_names) != self.channel_count:
            raise IntegrationContractError(
                IntegrationErrorCode.SCHEMA_MISMATCH,
                "Channel names must match the temporal channel count.",
                context={
                    "channel_count": self.channel_count,
                    "channel_names": list(self.channel_names),
                },
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_count": self.series_count,
            "channel_count": self.channel_count,
            "history_length": self.history_length,
            "orientation": self.orientation.value,
            "source_shape": list(self.source_shape),
            "channel_names": list(self.channel_names),
        }


@dataclass(frozen=True)
class ForecastingDataPlan:
    """Pure conversion plan for history supplied to a forecaster."""

    profile: DataProfile
    stage: DataStage
    horizon: int
    schema: TemporalSchema

    def __post_init__(self) -> None:
        if self.profile is not DataProfile.TENSOR:
            raise IntegrationContractError(
                IntegrationErrorCode.UNKNOWN_PROFILE,
                "Forecasting integration requires the TensorData profile.",
                context={"profile": self.profile.value},
            )
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_HORIZON,
                "Forecast horizon must be a positive integer.",
                context={"horizon": self.horizon},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "stage": self.stage.value,
            "horizon": self.horizon,
            "schema": self.schema.to_dict(),
        }


@dataclass(frozen=True)
class PreparedForecastingData:
    """Canonical FEDOT-wide history with separate object and time coordinates."""

    values: np.ndarray
    sample_idx: np.ndarray
    history_time_idx: np.ndarray
    plan: ForecastingDataPlan

    def __post_init__(self) -> None:
        values = _readonly_array(self.values, "values")
        sample_idx = _readonly_array(
            self.sample_idx, "sample_idx", expected_rank=1)
        time_idx = _readonly_array(
            self.history_time_idx, "history_time_idx", expected_rank=1)
        expected_shape = _canonical_shape(self.plan.schema)
        if values.shape != expected_shape:
            raise IntegrationContractError(
                IntegrationErrorCode.SCHEMA_MISMATCH,
                "Canonical forecasting values do not match the plan.",
                context={"expected": list(
                    expected_shape), "observed": list(values.shape)},
            )
        if len(sample_idx) != self.plan.schema.series_count:
            raise IntegrationContractError(
                IntegrationErrorCode.INDEX_MISMATCH,
                "Forecasting sample coordinates must match the series count.",
                context={"samples": len(
                    sample_idx), "series_count": self.plan.schema.series_count},
            )
        if len(time_idx) != self.plan.schema.history_length:
            raise IntegrationContractError(
                IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
                "History time coordinates must match the temporal length.",
                context={"coordinates": len(
                    time_idx), "history_length": self.plan.schema.history_length},
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_idx", sample_idx)
        object.__setattr__(self, "history_time_idx", time_idx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values_shape": list(self.values.shape),
            "sample_idx": _jsonable(self.sample_idx),
            "history_time_idx": _jsonable(self.history_time_idx),
            "plan": self.plan.to_dict(),
        }


@dataclass(frozen=True)
class ForecastingPrediction:
    """Forecast values with explicit future coordinates."""

    values: np.ndarray
    forecast_time_idx: np.ndarray
    sample_idx: np.ndarray
    horizon: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _readonly_array(self.values, "values")
        future_idx = _readonly_array(
            self.forecast_time_idx, "forecast_time_idx", expected_rank=1)
        sample_idx = _readonly_array(
            self.sample_idx, "sample_idx", expected_rank=1)
        if values.ndim not in (1, 2) or self.horizon < 1 or values.shape[-1] != self.horizon:
            raise IntegrationContractError(
                IntegrationErrorCode.LENGTH_MISMATCH,
                "Forecast output must be one- or two-dimensional and contain the requested horizon.",
                context={"horizon": self.horizon, "shape": list(values.shape)},
            )
        if len(future_idx) != self.horizon:
            raise IntegrationContractError(
                IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
                "Future time coordinates must match the forecast horizon.",
                context={"horizon": self.horizon,
                         "coordinates": len(future_idx)},
            )
        expected_series = 1 if values.ndim == 1 else values.shape[0]
        if len(sample_idx) != expected_series:
            raise IntegrationContractError(
                IntegrationErrorCode.INDEX_MISMATCH,
                "Forecast sample coordinates must match the prediction series count.",
                context={"samples": len(sample_idx),
                         "series_count": expected_series},
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "forecast_time_idx", future_idx)
        object.__setattr__(self, "sample_idx", sample_idx)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": _jsonable(self.values),
            "forecast_time_idx": _jsonable(self.forecast_time_idx),
            "sample_idx": _jsonable(self.sample_idx),
            "horizon": self.horizon,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class TemporalModalityPlan:
    """One named signal and the period for which it must be available."""

    name: str
    role: TemporalSignalRole
    schema: TemporalSchema
    required_future_steps: int = 0

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Temporal modality name must not be empty.",
            )
        if self.required_future_steps < 0:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_HORIZON,
                "Required future steps cannot be negative.",
                context={"modality": name,
                         "steps": self.required_future_steps},
            )
        if self.role is not TemporalSignalRole.KNOWN_FUTURE and self.required_future_steps:
            raise IntegrationContractError(
                IntegrationErrorCode.FUTURE_LEAKAGE,
                "Only known-future signals may provide forecast-period values.",
                context={"modality": name, "role": self.role.value},
            )
        object.__setattr__(self, "name", name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "required_future_steps": self.required_future_steps,
            "schema": self.schema.to_dict(),
        }


@dataclass(frozen=True)
class TemporalMultimodalPlan:
    """Deterministic collection of aligned temporal modalities."""

    horizon: int
    modalities: Mapping[str, TemporalModalityPlan]

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_HORIZON,
                "Multimodal forecast horizon must be positive.",
                context={"horizon": self.horizon},
            )
        values = dict(self.modalities)
        if not values or not any(item.role is TemporalSignalRole.TARGET for item in values.values()):
            raise IntegrationContractError(
                IntegrationErrorCode.MODALITY_MISMATCH,
                "Temporal multimodal data requires a target signal.",
                context={"modalities": sorted(values)},
            )
        for name, item in values.items():
            if name != item.name:
                raise IntegrationContractError(
                    IntegrationErrorCode.MODALITY_MISMATCH,
                    "Temporal modality keys must match their declared names.",
                    context={"key": name, "declared_name": item.name},
                )
            if item.role is TemporalSignalRole.KNOWN_FUTURE and item.required_future_steps < self.horizon:
                raise IntegrationContractError(
                    IntegrationErrorCode.FUTURE_LEAKAGE,
                    "Known-future signals must cover the complete forecast horizon.",
                    context={"modality": name, "required": self.horizon,
                             "available": item.required_future_steps},
                )
        object.__setattr__(self, "modalities", MappingProxyType(
            dict(sorted(values.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "modalities": {name: item.to_dict() for name, item in self.modalities.items()},
        }


@dataclass(frozen=True)
class AnomalyInterval:
    """Canonical half-open anomaly interval in positional coordinates."""

    start: int
    stop: int
    label: str = "anomaly"

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_INTERVAL,
                "Anomaly intervals must be non-empty and half-open.",
                context={"start": self.start, "stop": self.stop},
            )
        if not self.label.strip():
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_INTERVAL,
                "Anomaly interval label must not be empty.",
            )

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "stop": self.stop, "label": self.label}


@dataclass(frozen=True)
class DetectionDataPlan:
    """Pure point-level anomaly-detection data contract."""

    profile: DataProfile
    stage: DataStage
    mode: DetectionMode
    schema: TemporalSchema
    causal: bool = True

    def __post_init__(self) -> None:
        if self.profile is not DataProfile.TENSOR:
            raise IntegrationContractError(
                IntegrationErrorCode.UNKNOWN_PROFILE,
                "Detection integration requires the TensorData profile.",
                context={"profile": self.profile.value},
            )
        if self.schema.series_count != 1:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "One detection request must describe one aligned signal.",
                context={"series_count": self.schema.series_count},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "stage": self.stage.value,
            "mode": self.mode.value,
            "causal": self.causal,
            "schema": self.schema.to_dict(),
        }


@dataclass(frozen=True)
class PreparedDetectionData:
    """Canonical point-by-channel signal with aligned interval labels."""

    values: np.ndarray
    time_idx: np.ndarray
    intervals: tuple[AnomalyInterval, ...]
    plan: DetectionDataPlan
    gap_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = _readonly_array(self.values, "values")
        time_idx = _readonly_array(self.time_idx, "time_idx", expected_rank=1)
        expected = (self.plan.schema.history_length,
                    self.plan.schema.channel_count)
        if values.shape != expected:
            raise IntegrationContractError(
                IntegrationErrorCode.SCHEMA_MISMATCH,
                "Detection values must use point-by-channel orientation.",
                context={"expected": list(
                    expected), "observed": list(values.shape)},
            )
        if len(time_idx) != expected[0]:
            raise IntegrationContractError(
                IntegrationErrorCode.TEMPORAL_INDEX_MISMATCH,
                "Detection coordinates must match the point count.",
                context={"points": expected[0], "coordinates": len(time_idx)},
            )
        for interval in self.intervals:
            if interval.stop > expected[0]:
                raise IntegrationContractError(
                    IntegrationErrorCode.INVALID_INTERVAL,
                    "An anomaly interval exceeds the signal length.",
                    context={"interval": interval.to_dict(),
                             "points": expected[0]},
                )
        gap_mask = None
        if self.gap_mask is not None:
            gap_mask = _readonly_array(
                self.gap_mask, "gap_mask", expected_rank=1)
            if len(gap_mask) != expected[0]:
                raise IntegrationContractError(
                    IntegrationErrorCode.LENGTH_MISMATCH,
                    "Gap mask must match the detection point count.",
                    context={"mask": len(gap_mask), "points": expected[0]},
                )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_idx", time_idx)
        object.__setattr__(self, "gap_mask", gap_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values_shape": list(self.values.shape),
            "time_idx": _jsonable(self.time_idx),
            "intervals": [interval.to_dict() for interval in self.intervals],
            "gap_mask": None if self.gap_mask is None else _jsonable(self.gap_mask),
            "plan": self.plan.to_dict(),
        }


@dataclass(frozen=True)
class DetectionPrediction:
    """Point-level detector output with coordinates and canonical events."""

    values: np.ndarray
    time_idx: np.ndarray
    mode: DetectionMode
    intervals: tuple[AnomalyInterval, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _readonly_array(self.values, "values")
        time_idx = _readonly_array(self.time_idx, "time_idx", expected_rank=1)
        expected_rank = 2 if self.mode is DetectionMode.PROBABILITIES else 1
        if values.ndim != expected_rank or (
                self.mode is DetectionMode.PROBABILITIES and values.shape[1] != 2
        ):
            raise IntegrationContractError(
                IntegrationErrorCode.SCHEMA_MISMATCH,
                "Detection output shape does not match its requested mode.",
                context={"mode": self.mode.value, "shape": list(values.shape)},
            )
        if values.shape[0] != len(time_idx):
            raise IntegrationContractError(
                IntegrationErrorCode.LENGTH_MISMATCH,
                "Detection output and time coordinates must have equal length.",
                context={"values": values.shape[0],
                         "coordinates": len(time_idx)},
            )
        for interval in self.intervals:
            if interval.stop > len(time_idx):
                raise IntegrationContractError(
                    IntegrationErrorCode.INVALID_INTERVAL,
                    "A predicted anomaly interval exceeds the output length.",
                    context={"interval": interval.to_dict(),
                             "points": len(time_idx)},
                )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_idx", time_idx)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": _jsonable(self.values),
            "time_idx": _jsonable(self.time_idx),
            "mode": self.mode.value,
            "intervals": [interval.to_dict() for interval in self.intervals],
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class TemporalRuntimeSnapshot:
    """Serializable lifecycle state for a temporal runtime."""

    profile: DataProfile
    state: RuntimeState
    operation_name: str
    plan: ForecastingDataPlan | DetectionDataPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "state": self.state.value,
            "operation_name": self.operation_name,
            "plan": None if self.plan is None else self.plan.to_dict(),
        }


def _canonical_shape(schema: TemporalSchema) -> tuple[int, ...]:
    if schema.channel_count == 1:
        return schema.series_count, schema.history_length
    return schema.series_count, schema.channel_count, schema.history_length


def _readonly_array(value: Any, name: str, expected_rank: int | None = None) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.ndim == 0:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            f"{name} must contain an explicit axis.",
            context={"field": name},
        )
    if expected_rank is not None and array.ndim != expected_rank:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            f"{name} must have rank {expected_rank}.",
            context={"field": name, "rank": array.ndim},
        )
    array.setflags(write=False)
    return array


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    return str(value)

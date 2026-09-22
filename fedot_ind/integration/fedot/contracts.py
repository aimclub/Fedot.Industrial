"""Typed values shared by the pure FEDOT integration core."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class DataProfile(str, Enum):
    """Supported representations at the Industrial/FEDOT boundary."""

    LEGACY = "legacy"
    TENSOR = "tensor"


class DataStage(str, Enum):
    """Lifecycle stage for which input data is prepared."""

    TRAIN = "train"
    PREDICT = "predict"


class IntegrationTask(str, Enum):
    """FEDOT task identifiers without importing FEDOT runtime types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "ts_forecasting"


class PredictionMode(str, Enum):
    """Supported model output representations at the integration boundary."""

    DEFAULT = "default"
    LABELS = "labels"
    PROBABILITIES = "probs"
    FULL_PROBABILITIES = "full_probs"


class IntegrationErrorCode(str, Enum):
    """Machine-readable categories for expected integration failures."""

    UNKNOWN_PROFILE = "unknown_profile"
    UNKNOWN_TASK = "unknown_task"
    UNKNOWN_STAGE = "unknown_stage"
    INVALID_AXES = "invalid_axes"
    INVALID_DATA = "invalid_data"
    SCHEMA_MISMATCH = "schema_mismatch"
    PLAN_MISMATCH = "plan_mismatch"
    LENGTH_MISMATCH = "length_mismatch"
    INVALID_STATE = "invalid_state"
    INDEX_MISMATCH = "index_mismatch"
    MODALITY_MISMATCH = "modality_mismatch"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    UNSUPPORTED_OUTPUT_MODE = "unsupported_output_mode"
    RUNTIME_FAILURE = "runtime_failure"
    INVALID_HORIZON = "invalid_horizon"
    INVALID_TEMPORAL_ORDER = "invalid_temporal_order"
    TEMPORAL_INDEX_MISMATCH = "temporal_index_mismatch"
    FUTURE_LEAKAGE = "future_leakage"
    INVALID_INTERVAL = "invalid_interval"


class RuntimeState(str, Enum):
    """Lifecycle of one effectful integration runtime."""

    CREATED = "created"
    FITTED = "fitted"
    CLOSED = "closed"


class IntegrationContractError(ValueError):
    """Expected contract failure with a stable code and serializable context."""

    def __init__(
            self,
            code: IntegrationErrorCode,
            message: str,
            *,
            context: Mapping[str, Any] | None = None,
            cause: Exception | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.cause = cause
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": _jsonable(self.context),
        }


@dataclass(frozen=True)
class AxisLayout:
    """Locations of semantic axes in source data."""

    sample: int
    feature: int | None = None
    channel: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "sample": self.sample,
            "feature": self.feature,
            "channel": self.channel,
        }


@dataclass(frozen=True)
class FeatureSchema:
    """Feature identity after axes are moved to their canonical positions."""

    dimensions: tuple[int, ...]
    columns: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": list(self.dimensions),
            "columns": None if self.columns is None else list(self.columns),
        }


@dataclass(frozen=True)
class DataPreparationPlan:
    """Serializable decision describing a pure input transformation."""

    profile: DataProfile
    task: IntegrationTask
    stage: DataStage
    source_rank: int
    axes: AxisLayout
    feature_schema: FeatureSchema

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "task": self.task.value,
            "stage": self.stage.value,
            "source_rank": self.source_rank,
            "axes": self.axes.to_dict(),
            "feature_schema": self.feature_schema.to_dict(),
        }


@dataclass(frozen=True)
class ModelExecutionPlan:
    """Validated model choice and immutable runtime parameters."""

    profile: DataProfile
    task: IntegrationTask
    operation_name: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, DataProfile):
            raise IntegrationContractError(
                IntegrationErrorCode.UNKNOWN_PROFILE,
                "Model execution profile must be a DataProfile value.",
                context={"profile": self.profile},
            )
        if not isinstance(self.task, IntegrationTask):
            raise IntegrationContractError(
                IntegrationErrorCode.UNKNOWN_TASK,
                "Model execution task must be an IntegrationTask value.",
                context={"task": self.task},
            )
        if self.task not in (IntegrationTask.CLASSIFICATION, IntegrationTask.REGRESSION):
            raise IntegrationContractError(
                IntegrationErrorCode.UNSUPPORTED_OPERATION,
                "The supervised runtime supports classification and regression only.",
                context={"task": self.task.value},
            )
        name = self.operation_name.strip()
        if not name:
            raise IntegrationContractError(
                IntegrationErrorCode.UNSUPPORTED_OPERATION,
                "Operation name must not be empty.",
                context={"operation": self.operation_name},
            )
        object.__setattr__(self, "operation_name", name)
        copied = deepcopy(dict(self.parameters))
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType({key: _freeze(value) for key, value in copied.items()}),
        )

    def runtime_parameters(self) -> dict[str, Any]:
        """Return an owned mutable copy for the effectful model constructor."""
        return {key: _thaw(value) for key, value in self.parameters.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "task": self.task.value,
            "operation_name": self.operation_name,
            "parameters": _jsonable(self.parameters),
        }


@dataclass(frozen=True)
class PreparedData:
    """Normalized values with the source sample index preserved."""

    values: np.ndarray
    idx: np.ndarray
    schema: FeatureSchema

    def __post_init__(self) -> None:
        values = _readonly_array(self.values, "values")
        idx = _readonly_array(self.idx, "idx", expected_rank=1)
        if values.shape[0] != idx.shape[0]:
            raise IntegrationContractError(
                IntegrationErrorCode.LENGTH_MISMATCH,
                "Prepared values and index must contain the same number of samples.",
                context={"samples": values.shape[0], "index_size": idx.shape[0]},
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "idx", idx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "idx": self.idx.tolist(),
            "schema": self.schema.to_dict(),
        }


@dataclass(frozen=True)
class MultimodalPreparationPlan:
    """Immutable per-modality plans for one train or predict request."""

    profile: DataProfile
    task: IntegrationTask
    stage: DataStage
    modalities: Mapping[str, DataPreparationPlan]

    def __post_init__(self) -> None:
        if self.profile is not DataProfile.TENSOR:
            raise IntegrationContractError(
                IntegrationErrorCode.UNKNOWN_PROFILE,
                "Multimodal preparation requires the TensorData profile.",
                context={"profile": self.profile.value},
            )
        values = dict(self.modalities)
        if not values or any(
                not isinstance(name, str) or not name.strip()
                for name in values
        ):
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Multimodal data must contain named modalities.",
                context={"modalities": list(values)},
            )
        normalized = dict(sorted(values.items()))
        for name, plan in normalized.items():
            if (
                    plan.profile is not self.profile
                    or plan.task is not self.task
                    or plan.stage is not self.stage
            ):
                raise IntegrationContractError(
                    IntegrationErrorCode.PLAN_MISMATCH,
                    "A modality plan does not match its multimodal parent plan.",
                    context={"modality": name, "plan": plan.to_dict()},
                )
        object.__setattr__(self, "modalities", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "task": self.task.value,
            "stage": self.stage.value,
            "modalities": {
                name: plan.to_dict() for name, plan in self.modalities.items()
            },
        }


@dataclass(frozen=True)
class PreparedMultimodalData:
    """Normalized modalities that share exact sample coordinates."""

    modalities: Mapping[str, PreparedData]

    def __post_init__(self) -> None:
        values = dict(self.modalities)
        if not values:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Prepared multimodal data must not be empty.",
            )
        if any(
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(value, PreparedData)
                for name, value in values.items()
        ):
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Prepared modalities must map non-empty names to PreparedData values.",
            )
        normalized = dict(sorted(values.items()))
        reference_name, reference = next(iter(normalized.items()))
        for name, modality in normalized.items():
            if not np.array_equal(reference.idx, modality.idx):
                raise IntegrationContractError(
                    IntegrationErrorCode.INDEX_MISMATCH,
                    "All modalities must use the same ordered sample index.",
                    context={
                        "reference_modality": reference_name,
                        "modality": name,
                        "reference_index": reference.idx.tolist(),
                        "index": modality.idx.tolist(),
                    },
                )
        object.__setattr__(self, "modalities", MappingProxyType(normalized))

    @property
    def idx(self) -> np.ndarray:
        return next(iter(self.modalities.values())).idx

    @property
    def sample_count(self) -> int:
        return int(self.idx.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "modalities": {
                name: prepared.to_dict()
                for name, prepared in self.modalities.items()
            }
        }


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Serializable runtime state without retaining model objects."""

    profile: DataProfile
    state: RuntimeState
    train_schema: FeatureSchema | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "state": self.state.value,
            "train_schema": None if self.train_schema is None else self.train_schema.to_dict(),
        }


@dataclass(frozen=True)
class PredictionBatch:
    """Immutable prediction values and their sample/class coordinates."""

    values: np.ndarray
    idx: np.ndarray
    classes: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = _readonly_array(self.values, "values")
        idx = _readonly_array(self.idx, "idx", expected_rank=1)
        classes = None
        if self.classes is not None:
            classes = _readonly_array(self.classes, "classes", expected_rank=1)
        if values.shape[0] != idx.shape[0]:
            raise IntegrationContractError(
                IntegrationErrorCode.LENGTH_MISMATCH,
                "Prediction values and index must contain the same number of samples.",
                context={"samples": values.shape[0], "index_size": idx.shape[0]},
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "classes", classes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "idx": self.idx.tolist(),
            "classes": None if self.classes is None else self.classes.tolist(),
        }


def _readonly_array(value: Any, name: str, expected_rank: int | None = None) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.ndim == 0:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            f"{name} must contain an explicit sample axis.",
            context={"field": name, "rank": array.ndim},
        )
    if expected_rank is not None and array.ndim != expected_rank:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            f"{name} must have rank {expected_rank}.",
            context={"field": name, "rank": array.ndim, "expected_rank": expected_rank},
        )
    array.setflags(write=False)
    return array


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly_array(value, "parameter")
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw(item) for item in value}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return deepcopy(value)

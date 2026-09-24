"""Thin effectful runtimes selected through the pure integration contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from fedot_ind.integration.fedot.contracts import (
    DataPreparationPlan,
    DataProfile,
    DataStage,
    IntegrationContractError,
    IntegrationErrorCode,
    IntegrationTask,
    ModelExecutionPlan,
    PredictionMode,
    PredictionBatch,
    PreparedData,
    RuntimeSnapshot,
    RuntimeState,
)
from fedot_ind.integration.fedot.data import normalize_input_data
from fedot_ind.integration.fedot.planning import SupportedData, build_data_plan, validate_prediction_plan


class RegressionRuntime(ABC):
    """One fitted regression model with explicit lifecycle ownership."""

    profile: DataProfile

    def __init__(self) -> None:
        self._state = RuntimeState.CREATED
        self._train_plan: DataPreparationPlan | None = None

    @property
    def snapshot(self) -> RuntimeSnapshot:
        schema = None if self._train_plan is None else self._train_plan.feature_schema
        return RuntimeSnapshot(self.profile, self._state, schema)

    def fit(self, features: SupportedData, target: Any) -> "RegressionRuntime":
        self._require_state(RuntimeState.CREATED, "fit")
        plan = build_data_plan(
            features,
            profile=self.profile,
            task=IntegrationTask.REGRESSION,
            stage=DataStage.TRAIN,
        )
        prepared = normalize_input_data(features, plan)
        target_values = _target_values(target, prepared.values.shape[0])
        try:
            self._fit(prepared, target_values)
        except IntegrationContractError:
            raise
        except Exception as error:
            raise _runtime_failure(self.profile, "fit", error) from error
        self._train_plan = plan
        self._state = RuntimeState.FITTED
        return self

    def predict(self, features: SupportedData) -> PredictionBatch:
        self._require_state(RuntimeState.FITTED, "predict")
        plan = build_data_plan(
            features,
            profile=self.profile,
            task=IntegrationTask.REGRESSION,
            stage=DataStage.PREDICT,
        )
        train_plan = self._train_plan
        if train_plan is None:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_STATE,
                "Fitted runtime has no training plan.",
                context={"state": self._state.value},
            )
        validate_prediction_plan(train_plan, plan)
        prepared = normalize_input_data(features, plan)
        try:
            values = self._predict(prepared)
        except IntegrationContractError:
            raise
        except Exception as error:
            raise _runtime_failure(self.profile, "predict", error) from error
        return PredictionBatch(values=np.asarray(values), idx=prepared.idx)

    def close(self) -> None:
        if self._state is RuntimeState.CLOSED:
            return
        try:
            self._close()
        except Exception as error:
            raise _runtime_failure(self.profile, "close", error) from error
        finally:
            self._train_plan = None
            self._state = RuntimeState.CLOSED

    def _require_state(self, expected: RuntimeState, action: str) -> None:
        if self._state is not expected:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_STATE,
                f"Cannot {action} an integration runtime in state {self._state.value!r}.",
                context={"action": action, "expected": expected.value, "actual": self._state.value},
            )

    @abstractmethod
    def _fit(self, data: PreparedData, target: np.ndarray) -> None:
        """Fit the profile-specific runtime."""

    @abstractmethod
    def _predict(self, data: PreparedData) -> np.ndarray:
        """Predict through the profile-specific runtime."""

    def _close(self) -> None:
        """Release profile-specific resources."""


class SupervisedRuntime(ABC):
    """One Industrial classification or regression operation with explicit state."""

    def __init__(self, plan: ModelExecutionPlan) -> None:
        self.plan = plan
        self.profile = plan.profile
        self._state = RuntimeState.CREATED
        self._train_plan: DataPreparationPlan | None = None
        self._classes: np.ndarray | None = None

    @property
    def snapshot(self) -> RuntimeSnapshot:
        schema = None if self._train_plan is None else self._train_plan.feature_schema
        return RuntimeSnapshot(self.profile, self._state, schema)

    def fit(self, features: SupportedData, target: Any) -> "SupervisedRuntime":
        self._require_state(RuntimeState.CREATED, "fit")
        plan = build_data_plan(
            features,
            profile=self.profile,
            task=self.plan.task,
            stage=DataStage.TRAIN,
        )
        prepared = normalize_input_data(features, plan)
        target_values = _supervised_target_values(
            target,
            expected_length=prepared.values.shape[0],
            task=self.plan.task,
        )
        try:
            self._fit(prepared, target_values)
        except IntegrationContractError:
            raise
        except Exception as error:
            raise _runtime_failure(self.profile, "fit", error) from error
        self._train_plan = plan
        if self.plan.task is IntegrationTask.CLASSIFICATION:
            self._classes = np.unique(target_values)
        self._state = RuntimeState.FITTED
        return self

    def predict(
            self,
            features: SupportedData,
            mode: PredictionMode | str = PredictionMode.DEFAULT,
    ) -> PredictionBatch:
        self._require_state(RuntimeState.FITTED, "predict")
        selected_mode = _prediction_mode(mode, self.plan.task)
        plan = build_data_plan(
            features,
            profile=self.profile,
            task=self.plan.task,
            stage=DataStage.PREDICT,
        )
        if self._train_plan is None:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_STATE,
                "Fitted runtime has no training plan.",
                context={"state": self._state.value},
            )
        validate_prediction_plan(self._train_plan, plan)
        prepared = normalize_input_data(features, plan)
        try:
            values = self._predict(prepared, selected_mode)
        except IntegrationContractError:
            raise
        except Exception as error:
            raise _runtime_failure(self.profile, "predict", error) from error
        classes = self._classes if self.plan.task is IntegrationTask.CLASSIFICATION else None
        normalized_values = np.asarray(values)
        if (
                classes is not None
                and selected_mode in (PredictionMode.DEFAULT, PredictionMode.LABELS)
        ):
            normalized_values = _decode_class_labels(normalized_values, classes)
        return PredictionBatch(values=normalized_values, idx=prepared.idx, classes=classes)

    def close(self) -> None:
        if self._state is RuntimeState.CLOSED:
            return
        try:
            self._close()
        except Exception as error:
            raise _runtime_failure(self.profile, "close", error) from error
        finally:
            self._train_plan = None
            self._classes = None
            self._state = RuntimeState.CLOSED

    def _require_state(self, expected: RuntimeState, action: str) -> None:
        if self._state is not expected:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_STATE,
                f"Cannot {action} an integration runtime in state {self._state.value!r}.",
                context={"action": action, "expected": expected.value, "actual": self._state.value},
            )

    @abstractmethod
    def _fit(self, data: PreparedData, target: np.ndarray) -> None:
        """Fit the operation selected by the execution plan."""

    @abstractmethod
    def _predict(self, data: PreparedData, mode: PredictionMode) -> np.ndarray:
        """Predict through the fitted operation."""

    def _close(self) -> None:
        """Release profile-specific resources."""


def create_regression_runtime(profile: DataProfile | str) -> RegressionRuntime:
    """Load the TensorData regression runtime without compatibility fallback."""
    selected = _profile(profile)
    try:
        runtime_type = getattr(
            import_module("fedot_ind.integration.fedot.tensor"),
            "TensorRegressionRuntime",
        )
        return runtime_type()
    except IntegrationContractError:
        raise
    except Exception as error:
        raise _runtime_failure(selected, "load", error) from error


def create_supervised_runtime(
        *,
        profile: DataProfile | str,
        task: IntegrationTask | str,
        operation_name: str,
        parameters: dict[str, Any] | None = None,
) -> SupervisedRuntime:
    """Create one tensor-era Industrial model runtime without implicit fallback."""
    selected_profile = _profile(profile)
    selected_task = _task(task)
    plan = ModelExecutionPlan(
        profile=selected_profile,
        task=selected_task,
        operation_name=operation_name,
        parameters=parameters or {},
    )
    _validate_supervised_operation(plan)
    try:
        runtime_type = getattr(
            import_module("fedot_ind.integration.fedot.tensor"),
            "TensorSupervisedRuntime",
        )
        return runtime_type(plan)
    except IntegrationContractError:
        raise
    except Exception as error:
        raise _runtime_failure(selected_profile, "load", error) from error


def create_multimodal_tensor_data(*args, **kwargs):
    """Load the TensorData multimodal bridge only when it is requested."""
    try:
        bridge = getattr(
            import_module("fedot_ind.integration.fedot.tensor"),
            "create_multimodal_tensor_data",
        )
        return bridge(*args, **kwargs)
    except IntegrationContractError:
        raise
    except Exception as error:
        raise _runtime_failure(DataProfile.TENSOR, "multimodal_bridge", error) from error


def create_forecasting_runtime(*args, **kwargs):
    """Load the TensorData forecasting runtime only when requested."""
    runtime_factory = getattr(
        import_module("fedot_ind.integration.fedot.forecasting"),
        "create_forecasting_runtime",
    )
    return runtime_factory(*args, **kwargs)


def create_detection_runtime(*args, **kwargs):
    """Load the TensorData detection runtime only when requested."""
    runtime_factory = getattr(
        import_module("fedot_ind.integration.fedot.detection"),
        "create_detection_runtime",
    )
    return runtime_factory(*args, **kwargs)


def _validate_supervised_operation(plan: ModelExecutionPlan) -> None:
    from fedot_ind.integration.fedot.extensions.catalog import (
        load_industrial_extension_catalog,
    )
    from fedot_ind.integration.fedot.extensions.contracts import IndustrialOperationKind

    declaration = next(
        (
            operation
            for operation in load_industrial_extension_catalog().operations
            if operation.name == plan.operation_name
        ),
        None,
    )
    if (
            declaration is None
            or declaration.kind is not IndustrialOperationKind.MODEL
            or plan.task.value not in declaration.tasks
    ):
        raise IntegrationContractError(
            IntegrationErrorCode.UNSUPPORTED_OPERATION,
            "Industrial operation is not available for the requested task.",
            context={
                "operation": plan.operation_name,
                "task": plan.task.value,
            },
        )


def _target_values(target: Any, expected_length: int) -> np.ndarray:
    if isinstance(target, (pd.Series, pd.DataFrame)):
        values = target.to_numpy(copy=True)
    else:
        values = np.array(target, copy=True)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            "Regression target must be one-dimensional.",
            context={"shape": list(values.shape)},
        )
    if values.shape[0] != expected_length:
        raise IntegrationContractError(
            IntegrationErrorCode.LENGTH_MISMATCH,
            "Features and target must contain the same number of samples.",
            context={"samples": expected_length, "target_size": values.shape[0]},
        )
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _supervised_target_values(
        target: Any,
        *,
        expected_length: int,
        task: IntegrationTask,
) -> np.ndarray:
    values = _target_values(target, expected_length)
    if task is IntegrationTask.CLASSIFICATION:
        if len(np.unique(values)) < 2:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_DATA,
                "Classification target must contain at least two classes.",
                context={"classes": np.unique(values).tolist()},
            )
        return values
    try:
        return values.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.INVALID_DATA,
            "Regression target must contain numeric values.",
            context={"dtype": str(values.dtype)},
            cause=error,
        ) from error


def _profile(value: DataProfile | str) -> DataProfile:
    try:
        return value if isinstance(value, DataProfile) else DataProfile(value)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.UNKNOWN_PROFILE,
            "Unsupported FEDOT integration profile.",
            context={"profile": value, "allowed": [item.value for item in DataProfile]},
            cause=error,
        ) from error


def _task(value: IntegrationTask | str) -> IntegrationTask:
    try:
        return value if isinstance(value, IntegrationTask) else IntegrationTask(value)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.UNKNOWN_TASK,
            "Unsupported integration task.",
            context={"task": value, "allowed": [item.value for item in IntegrationTask]},
            cause=error,
        ) from error


def _prediction_mode(value: PredictionMode | str, task: IntegrationTask) -> PredictionMode:
    try:
        mode = value if isinstance(value, PredictionMode) else PredictionMode(value)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.UNSUPPORTED_OUTPUT_MODE,
            "Unsupported prediction output mode.",
            context={"mode": value, "allowed": [item.value for item in PredictionMode]},
            cause=error,
        ) from error
    if task is IntegrationTask.REGRESSION and mode not in (
            PredictionMode.DEFAULT,
            PredictionMode.LABELS,
    ):
        raise IntegrationContractError(
            IntegrationErrorCode.UNSUPPORTED_OUTPUT_MODE,
            "Regression does not provide class probabilities.",
            context={"mode": mode.value, "task": task.value},
        )
    return PredictionMode.DEFAULT if task is IntegrationTask.REGRESSION else mode


def _decode_class_labels(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    try:
        encoded = flat.astype(int)
    except (TypeError, ValueError):
        return values
    if not np.all(flat == encoded):
        return values
    if np.any(encoded < 0) or np.any(encoded >= len(classes)):
        return values
    return classes[encoded].reshape(np.asarray(values).shape)


def _runtime_failure(profile: DataProfile, phase: str, error: Exception) -> IntegrationContractError:
    return IntegrationContractError(
        IntegrationErrorCode.RUNTIME_FAILURE,
        f"FEDOT {profile.value!r} runtime failed during {phase}.",
        context={"profile": profile.value, "phase": phase, "cause_type": type(error).__name__},
        cause=error,
    )

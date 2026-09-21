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


def create_regression_runtime(profile: DataProfile | str) -> RegressionRuntime:
    """Load exactly one FEDOT profile implementation without fallback."""
    try:
        selected = profile if isinstance(profile, DataProfile) else DataProfile(profile)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.UNKNOWN_PROFILE,
            "Unsupported FEDOT integration profile.",
            context={"profile": profile, "allowed": [item.value for item in DataProfile]},
            cause=error,
        ) from error
    module_name, class_name = {
        DataProfile.LEGACY: ("fedot_ind.integration.fedot.legacy", "LegacyRegressionRuntime"),
        DataProfile.TENSOR: ("fedot_ind.integration.fedot.tensor", "TensorRegressionRuntime"),
    }[selected]
    try:
        runtime_type = getattr(import_module(module_name), class_name)
        return runtime_type()
    except IntegrationContractError:
        raise
    except Exception as error:
        raise _runtime_failure(selected, "load", error) from error


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


def _runtime_failure(profile: DataProfile, phase: str, error: Exception) -> IntegrationContractError:
    return IntegrationContractError(
        IntegrationErrorCode.RUNTIME_FAILURE,
        f"FEDOT {profile.value!r} runtime failed during {phase}.",
        context={"profile": profile.value, "phase": phase, "cause_type": type(error).__name__},
        cause=error,
    )

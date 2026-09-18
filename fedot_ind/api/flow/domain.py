"""Typed domain objects for Industrial API runtime planning."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

from pymonad.either import Either, Left, Right


@dataclass(frozen=True)
class IndustrialFlowError:
    """Expected functional-flow error represented as data."""

    code: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        if self.value is None:
            return f"{self.code}: {self.message}"
        return f"{self.code}: {self.message} ({self.value!r})"


class IndustrialTaskKind(str, Enum):
    """Closed set of task kinds supported by ``FedotIndustrial``."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TS_FORECASTING = "ts_forecasting"

    @property
    def is_forecasting(self) -> bool:
        return self is IndustrialTaskKind.TS_FORECASTING

    @property
    def is_regression_like(self) -> bool:
        return self in {IndustrialTaskKind.REGRESSION, IndustrialTaskKind.TS_FORECASTING}

    @classmethod
    def normalize(cls, value: Any) -> Either:
        if isinstance(value, cls):
            return Right(value)
        raw_value = str(value).strip().lower() if value is not None else ""
        aliases = {
            "classification": cls.CLASSIFICATION,
            "ts_classification": cls.CLASSIFICATION,
            "regression": cls.REGRESSION,
            "ts_regression": cls.REGRESSION,
            "forecasting": cls.TS_FORECASTING,
            "ts_forecasting": cls.TS_FORECASTING,
        }
        task_kind = aliases.get(raw_value)
        if task_kind is not None:
            return Right(task_kind)
        return Left(
            IndustrialFlowError(
                code="invalid_task_kind",
                message="Unsupported Industrial task kind",
                value=value,
            )
        )


class InitialAssumptionSource(str, Enum):
    """Source selected for the initial assumption value."""

    AUTOML_CONFIG = "automl_config"
    INDUSTRIAL_CONFIG = "industrial_config"
    NONE = "none"


@dataclass(frozen=True)
class InitialAssumptionPlan:
    """Resolved initial-assumption source and raw assumption value."""

    source: InitialAssumptionSource
    assumption: Any = None

    @property
    def has_assumption(self) -> bool:
        return self.source is not InitialAssumptionSource.NONE


def plan_initial_assumption(automl_initial: Any = None, industrial_initial: Any = None) -> InitialAssumptionPlan:
    """Choose the initial assumption source using current API precedence."""
    if automl_initial is not None:
        return InitialAssumptionPlan(InitialAssumptionSource.AUTOML_CONFIG, automl_initial)
    if industrial_initial is not None:
        return InitialAssumptionPlan(InitialAssumptionSource.INDUSTRIAL_CONFIG, industrial_initial)
    return InitialAssumptionPlan(InitialAssumptionSource.NONE, None)


class PredictionOutputMode(str, Enum):
    """Prediction output mode used by prediction routing."""

    DEFAULT = "default"
    LABELS = "labels"
    PROBABILITIES = "probs"

    @classmethod
    def normalize(cls, value: Optional[str]) -> Either:
        if isinstance(value, cls):
            return Right(value)
        raw_value = "default" if value is None else str(value).strip().lower()
        aliases = {
            "default": cls.DEFAULT,
            "labels": cls.LABELS,
            "label": cls.LABELS,
            "classes": cls.LABELS,
            "probs": cls.PROBABILITIES,
            "probabilities": cls.PROBABILITIES,
            "probability": cls.PROBABILITIES,
        }
        mode = aliases.get(raw_value)
        if mode is not None:
            return Right(mode)
        return Left(
            IndustrialFlowError(
                code="invalid_prediction_mode",
                message="Unsupported prediction output mode",
                value=value,
            )
        )


@dataclass(frozen=True)
class PredictionModePlan:
    """Resolved prediction mode together with task context."""

    task_kind: IndustrialTaskKind
    output_mode: PredictionOutputMode = PredictionOutputMode.DEFAULT

    @property
    def requires_probabilities(self) -> bool:
        return self.output_mode is PredictionOutputMode.PROBABILITIES


@dataclass(frozen=True)
class ProcessedInputBundle:
    """Input processing result passed from the input service to the facade."""

    data: Any
    target_encoder: Any = None
    fit_stage: bool = True


@dataclass(frozen=True)
class SolverInitPlan:
    """Typed data needed to construct a FEDOT solver."""

    task_kind: IndustrialTaskKind
    learning_strategy_params: Mapping[str, Any] = field(default_factory=dict)
    optimisation_loss: Any = None
    task_params: Any = None
    optimizer: Any = None
    available_operations: Tuple[str, ...] = ()
    initial_assumption: Any = None

    @classmethod
    def create(
            cls,
            *,
            task_kind: IndustrialTaskKind,
            learning_strategy_params: Optional[Mapping[str, Any]] = None,
            optimisation_loss: Any = None,
            task_params: Any = None,
            optimizer: Any = None,
            available_operations: Optional[Sequence[str]] = None,
            initial_assumption: Any = None,
    ) -> "SolverInitPlan":
        return cls(
            task_kind=task_kind,
            learning_strategy_params=dict(learning_strategy_params or {}),
            optimisation_loss=optimisation_loss,
            task_params=task_params,
            optimizer=optimizer,
            available_operations=tuple(available_operations or ()),
            initial_assumption=initial_assumption,
        )

    @property
    def fedot_problem(self) -> str:
        return self.task_kind.value

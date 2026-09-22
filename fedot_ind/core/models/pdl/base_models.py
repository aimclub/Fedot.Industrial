"""Lazy base-estimator registry owned by the PDL package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class BaseModelSpec:
    """Import location for one PDL downstream estimator."""

    module: str
    attribute: str

    def load(self):
        """Import the estimator only when the corresponding model is selected."""
        return getattr(import_module(self.module), self.attribute)


_CLASSIFICATION_MODELS: Mapping[str, BaseModelSpec] = MappingProxyType(
    {
        "xgboost": BaseModelSpec("sklearn.ensemble", "GradientBoostingClassifier"),
        "catboost": BaseModelSpec(
            "fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations",
            "FedotCatBoostClassificationImplementation",
        ),
        "logit": BaseModelSpec("sklearn.linear_model", "LogisticRegression"),
        "dt": BaseModelSpec("sklearn.tree", "DecisionTreeClassifier"),
        "rf": BaseModelSpec("sklearn.ensemble", "RandomForestClassifier"),
        "mlp": BaseModelSpec("sklearn.neural_network", "MLPClassifier"),
        "lgbm": BaseModelSpec("lightgbm.sklearn", "LGBMClassifier"),
    }
)

_REGRESSION_MODELS: Mapping[str, BaseModelSpec] = MappingProxyType(
    {
        "xgbreg": BaseModelSpec("xgboost", "XGBRegressor"),
        "sgdr": BaseModelSpec("sklearn.linear_model", "SGDRegressor"),
        "treg": BaseModelSpec("sklearn.ensemble", "ExtraTreesRegressor"),
        "ridge": BaseModelSpec("sklearn.linear_model", "Ridge"),
        "lasso": BaseModelSpec("sklearn.linear_model", "Lasso"),
        "dtreg": BaseModelSpec("sklearn.tree", "DecisionTreeRegressor"),
        "lgbmreg": BaseModelSpec("lightgbm.sklearn", "LGBMRegressor"),
        "catboostreg": BaseModelSpec(
            "fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations",
            "FedotCatBoostRegressionImplementation",
        ),
    }
)


def available_pdl_base_models(task: str) -> tuple[str, ...]:
    """Return stable model names for a PDL task."""
    return tuple(_registry_for(task))


def create_pdl_base_model(
    task: str,
    name: str,
    parameters: Mapping[str, Any] | None = None,
):
    """Instantiate one PDL downstream estimator through its lazy specification."""
    registry = _registry_for(task)
    try:
        spec = registry[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown PDL {task} base model {name!r}. "
            f"Available models: {', '.join(registry)}."
        ) from error
    return spec.load()(**dict(parameters or {}))


def _registry_for(task: str) -> Mapping[str, BaseModelSpec]:
    if task == "classification":
        return _CLASSIFICATION_MODELS
    if task == "regression":
        return _REGRESSION_MODELS
    raise ValueError(
        f"Unknown PDL task {task!r}. Available tasks: classification, regression."
    )

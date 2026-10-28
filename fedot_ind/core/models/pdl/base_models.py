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
    seed_parameter: str | None = 'random_state'

    def load(self):
        """Import the estimator only when the corresponding model is selected."""
        return getattr(import_module(self.module), self.attribute)


_CLASSIFICATION_MODELS: Mapping[str, BaseModelSpec] = MappingProxyType(
    {
        "xgboost": BaseModelSpec("sklearn.ensemble", "GradientBoostingClassifier"),
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
    }
)


def available_pdl_base_models(task: str) -> tuple[str, ...]:
    """Return stable model names for a PDL task."""
    return tuple(_registry_for(task))


def create_pdl_base_model(
    task: str,
    name: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    random_state: int | None = None,
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
    resolved = dict(parameters or {})
    if random_state is not None:
        if spec.seed_parameter is None:
            raise ValueError(f'PDL base model {name!r} does not support the seeded array estimator contract')
        if spec.seed_parameter in resolved and resolved[spec.seed_parameter] != random_state:
            raise ValueError('Conflicting random_state values for the PDL base model')
        resolved[spec.seed_parameter] = random_state
    return spec.load()(**resolved)


def resolved_base_model_parameters(model) -> dict[str, Any]:
    """Record actual estimator parameters separately from requested PDL inputs."""
    return {'base_model_class': f'{type(model).__module__}.{type(model).__qualname__}',
            'resolved_base_model_params': model.get_params(deep=False)}


def _registry_for(task: str) -> Mapping[str, BaseModelSpec]:
    if task == "classification":
        return _CLASSIFICATION_MODELS
    if task == "regression":
        return _REGRESSION_MODELS
    raise ValueError(
        f"Unknown PDL task {task!r}. Available tasks: classification, regression."
    )

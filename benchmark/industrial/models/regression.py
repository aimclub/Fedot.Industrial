from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmark.industrial.core import ModelSpec, RunStatus
from benchmark.industrial.errors import BenchmarkRegressionError
from benchmark.industrial.models.kernel_artifacts import export_kernel_learning_artifacts


@dataclass
class MeanRegressor:
    name: str = 'MeanRegressor'
    tags: tuple[str, ...] = ('baseline', 'regression')
    optional: bool = False
    mean_: float = 0.0

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        del features
        self.mean_ = float(np.mean(target))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean_, dtype=float)


@dataclass
class LinearRegressor:
    name: str = 'LinearRegressor'
    tags: tuple[str, ...] = ('baseline', 'regression')
    optional: bool = False
    coefficients_: np.ndarray | None = None

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        design = np.hstack([features, np.ones((features.shape[0], 1))])
        self.coefficients_, *_ = np.linalg.lstsq(design, target, rcond=None)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise BenchmarkRegressionError('LinearRegressor must be fitted before prediction.')
        design = np.hstack([features, np.ones((features.shape[0], 1))])
        return design @ self.coefficients_


@dataclass
class OptionalExternalRegressor:
    dependency_name: str
    name: str
    tags: tuple[str, ...] = ('baseline', 'regression', 'external')
    optional: bool = True

    def availability(self) -> tuple[RunStatus, str]:
        try:
            __import__(self.dependency_name)
            return RunStatus.SKIPPED, 'Adapter scaffold registered but training backend is not wired yet.'
        except Exception:
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'


@dataclass
class KernelEnsembleRegressorAdapter:
    name: str
    tags: tuple[str, ...] = ('industrial', 'regression', 'kernel_learning')
    optional: bool = False
    params: dict[str, Any] | None = None
    model_: Any | None = None

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from fedot_ind.core.kernel_learning import KernelEnsembleRegressor  # noqa: F401
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover
            return RunStatus.NOT_AVAILABLE, f'Kernel ensemble regressor is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from fedot_ind.core.kernel_learning import KernelEnsembleRegressor

        self.model_ = KernelEnsembleRegressor(**(self.params or {}))
        self.model_.fit(features, target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise BenchmarkRegressionError('KernelEnsembleRegressorAdapter must be fitted before prediction.')
        return self.model_.predict(features)

    def export_artifacts(self) -> dict[str, Any]:
        return export_kernel_learning_artifacts(self.model_)


def build_regression_model(spec: ModelSpec):
    name = spec.adapter_name.lower()
    if name == 'mean_regressor':
        return MeanRegressor(name=spec.display_name, tags=spec.tags or ('baseline', 'regression'))
    if name == 'linear_regressor':
        return LinearRegressor(name=spec.display_name, tags=spec.tags or ('baseline', 'regression'))
    if name == 'kernel_ensemble_regressor':
        return KernelEnsembleRegressorAdapter(
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'regression', 'kernel_learning'),
            optional=spec.optional,
            params=dict(spec.params),
        )
    if name == 'fedot_industrial_regressor':
        return OptionalExternalRegressor(
            dependency_name='fedot',
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'regression', 'external'),
        )
    raise BenchmarkRegressionError(f'Unsupported regression model adapter: {spec.adapter_name}')


__all__ = [
    "KernelEnsembleRegressorAdapter",
    "LinearRegressor",
    "MeanRegressor",
    "OptionalExternalRegressor",
    "build_regression_model",
]

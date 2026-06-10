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


@dataclass
class PDLRegressorAdapter:
    name: str
    tags: tuple[str, ...] = ('industrial', 'regression', 'pdl')
    optional: bool = True
    params: dict[str, Any] | None = None
    model_: Any | None = None

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from fedot.core.data.data import InputData  # noqa: F401
            from fedot.core.operations.operation_parameters import OperationParameters  # noqa: F401
            from fedot.core.repository.dataset_types import DataTypesEnum  # noqa: F401
            from fedot.core.repository.tasks import Task, TaskTypesEnum  # noqa: F401
            from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceRegressor  # noqa: F401
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover - optional FEDOT runtime boundary
            return RunStatus.NOT_AVAILABLE, f'PDL regressor is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceRegressor

        input_data = _fedot_input_data(features=features, target=np.asarray(target, dtype=float), task_type='regression')
        self.model_ = PairwiseDifferenceRegressor(params=_operation_parameters(self.params, default_model='treg'))
        self.model_.fit(input_data)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise BenchmarkRegressionError('PDLRegressorAdapter must be fitted before prediction.')
        dummy_target = np.zeros(features.shape[0], dtype=float)
        input_data = _fedot_input_data(features=features, target=dummy_target, task_type='regression')
        prediction = self.model_.predict(input_data)
        values = getattr(prediction, 'predict', prediction)
        return np.asarray(values, dtype=float).reshape(-1)


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
    if name in {'pdl_regressor', 'pdl_reg'}:
        return PDLRegressorAdapter(
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'regression', 'pdl'),
            optional=True,
            params=dict(spec.params),
        )
    if name == 'fedot_industrial_regressor':
        return OptionalExternalRegressor(
            dependency_name='fedot',
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'regression', 'external'),
        )
    raise BenchmarkRegressionError(f'Unsupported regression model adapter: {spec.adapter_name}')


def _operation_parameters(params: dict[str, Any] | None, *, default_model: str):
    from fedot.core.operations.operation_parameters import OperationParameters

    payload = {'model': default_model}
    payload.update(dict(params or {}))
    return OperationParameters(payload)


def _fedot_input_data(features: np.ndarray, target: np.ndarray, *, task_type: str):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    task = Task(TaskTypesEnum.classification if task_type == 'classification' else TaskTypesEnum.regression)
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=task,
        data_type=DataTypesEnum.table,
    )


__all__ = [
    "KernelEnsembleRegressorAdapter",
    "LinearRegressor",
    "MeanRegressor",
    "OptionalExternalRegressor",
    "PDLRegressorAdapter",
    "build_regression_model",
]

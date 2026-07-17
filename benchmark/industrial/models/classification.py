from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmark.industrial.core import ModelSpec, RunStatus
from benchmark.industrial.errors import BenchmarkClassificationError
from benchmark.industrial.models.kernel_artifacts import export_kernel_learning_artifacts


@dataclass
class MajorityClassClassifier:
    name: str = 'MajorityClass'
    tags: tuple[str, ...] = ('baseline', 'classification')
    optional: bool = False
    majority_label_: str = ''

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        del features
        values, counts = np.unique(target, return_counts=True)
        self.majority_label_ = str(values[np.argmax(counts)])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.majority_label_, dtype=object)


@dataclass
class NearestCentroidClassifier:
    name: str = 'NearestCentroid'
    tags: tuple[str, ...] = ('baseline', 'classification')
    optional: bool = False
    centroids_: dict[str, np.ndarray] | None = None

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self.centroids_ = {}
        for label in np.unique(target):
            self.centroids_[str(label)] = np.mean(features[target == label], axis=0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise BenchmarkClassificationError('NearestCentroidClassifier must be fitted before prediction.')
        predictions = []
        for row in features:
            label = min(
                self.centroids_.items(),
                key=lambda item: float(np.linalg.norm(row - item[1])),
            )[0]
            predictions.append(label)
        return np.asarray(predictions, dtype=object)


@dataclass
class OptionalExternalClassifier:
    dependency_name: str
    name: str
    tags: tuple[str, ...] = ('baseline', 'classification', 'external')
    optional: bool = True

    def availability(self) -> tuple[RunStatus, str]:
        try:
            __import__(self.dependency_name)
            return RunStatus.SKIPPED, 'Adapter scaffold registered but training backend is not wired yet.'
        except Exception:
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'


@dataclass
class KernelEnsembleClassifierAdapter:
    name: str
    tags: tuple[str, ...] = ('industrial', 'classification', 'kernel_learning')
    optional: bool = False
    params: dict[str, Any] | None = None
    model_: Any | None = None

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from fedot_ind.core.kernel_learning import KernelEnsembleClassifier  # noqa: F401
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover
            return RunStatus.NOT_AVAILABLE, f'Kernel ensemble classifier is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from fedot_ind.core.kernel_learning import KernelEnsembleClassifier

        self.model_ = KernelEnsembleClassifier(**(self.params or {}))
        self.model_.fit(features, target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise BenchmarkClassificationError('KernelEnsembleClassifierAdapter must be fitted before prediction.')
        return self.model_.predict(features)

    def export_artifacts(self) -> dict[str, Any]:
        return export_kernel_learning_artifacts(self.model_)


@dataclass
class PDLClassifierAdapter:
    name: str
    tags: tuple[str, ...] = ('industrial', 'classification', 'pdl')
    optional: bool = True
    params: dict[str, Any] | None = None
    model_: Any | None = None

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from fedot.core.data.data import InputData  # noqa: F401
            from fedot.core.operations.operation_parameters import OperationParameters  # noqa: F401
            from fedot.core.repository.dataset_types import DataTypesEnum  # noqa: F401
            from fedot.core.repository.tasks import Task, TaskTypesEnum  # noqa: F401
            from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceClassifier  # noqa: F401
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover - optional FEDOT runtime boundary
            return RunStatus.NOT_AVAILABLE, f'PDL classifier is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceClassifier

        input_data = _fedot_input_data(features=features, target=target, task_type='classification')
        self.model_ = PairwiseDifferenceClassifier(params=_operation_parameters(self.params, default_model='rf'))
        self.model_.fit(input_data)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise BenchmarkClassificationError('PDLClassifierAdapter must be fitted before prediction.')
        dummy_target = np.zeros(features.shape[0], dtype=int)
        input_data = _fedot_input_data(features=features, target=dummy_target, task_type='classification')
        prediction = self.model_.predict(input_data)
        values = getattr(prediction, 'predict', prediction)
        return np.asarray(values).reshape(-1).astype(object)


def build_classification_model(spec: ModelSpec):
    name = spec.adapter_name.lower()
    if name == 'majority_class':
        return MajorityClassClassifier(name=spec.display_name, tags=spec.tags or ('baseline', 'classification'))
    if name == 'nearest_centroid':
        return NearestCentroidClassifier(name=spec.display_name, tags=spec.tags or ('baseline', 'classification'))
    if name == 'kernel_ensemble_classifier':
        return KernelEnsembleClassifierAdapter(
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'kernel_learning'),
            optional=spec.optional,
            params=dict(spec.params),
        )
    if name in {'pdl_classifier', 'pdl_clf'}:
        return PDLClassifierAdapter(
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'pdl'),
            optional=True,
            params=dict(spec.params),
        )
    if name == 'fedot_industrial_classifier':
        return OptionalExternalClassifier(
            dependency_name='fedot',
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'external'),
        )
    raise BenchmarkClassificationError(f'Unsupported classification model adapter: {spec.adapter_name}')


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
    "KernelEnsembleClassifierAdapter",
    "MajorityClassClassifier",
    "NearestCentroidClassifier",
    "OptionalExternalClassifier",
    "PDLClassifierAdapter",
    "build_classification_model",
]

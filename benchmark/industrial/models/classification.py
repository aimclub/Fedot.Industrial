from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmark.industrial.core import ModelSpec, RunStatus
from benchmark.industrial.errors import BenchmarkClassificationError
from benchmark.industrial.models.kernel_artifacts import export_kernel_learning_artifacts
from fedot_ind.core.kernel_learning.generators.adapters import (
    BudgetedRepositoryFeatureGeneratorAdapter,
    GeneratorBudgetPolicy,
    OperationSpec,
    create_feature_generator,
)


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
class KernelFeatureGeneratorClassifierAdapter:
    name: str
    tags: tuple[str, ...] = ('industrial', 'classification', 'feature_generator')
    optional: bool = False
    params: dict[str, Any] | None = None
    generator_: Any | None = None
    model_: Any | None = None

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from sklearn.ensemble import RandomForestClassifier  # noqa: F401
            from sklearn.linear_model import LogisticRegression  # noqa: F401
            from sklearn.neighbors import KNeighborsClassifier  # noqa: F401
            from sklearn.svm import SVC  # noqa: F401
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover
            return RunStatus.NOT_AVAILABLE, f'Scikit-learn classifiers are unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        generator_name = (self.params or {}).get('generator_name', 'statistical_summary')
        generator_params = (self.params or {}).get('generator_params', {}) or {}
        classifier_name = (self.params or {}).get('classifier_name', 'logistic_regression')
        classifier_params = (self.params or {}).get('classifier_params', {}) or {}

        self.generator_ = self._build_feature_generator(generator_name, generator_params)
        transformed = self.generator_.fit_transform(features, target, task_type='classification')
        self.model_ = self._build_classifier(classifier_name, classifier_params)
        target_labels = np.asarray(target).reshape(-1).astype(str)
        self.model_.fit(transformed.features, target_labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.generator_ is None or self.model_ is None:
            raise BenchmarkClassificationError('KernelFeatureGeneratorClassifierAdapter must be fitted before prediction.')
        transformed = self.generator_.transform(features)
        return np.asarray(self.model_.predict(transformed.features), dtype=object)

    def _build_feature_generator(self, generator_name: str, generator_params: dict[str, Any]):

        generator = create_feature_generator(generator_name, torch_device='auto')

        if generator_params and hasattr(generator, 'operation_specs'):
            new_specs = []
            for spec in generator.operation_specs:
                merged_params = dict(spec.params)
                merged_params.update(generator_params)
                new_specs.append(OperationSpec(
                    name=spec.name,
                    module_path=spec.module_path,
                    class_name=spec.class_name,
                    params=merged_params,
                    use_torch=spec.use_torch,
                    fit_transform_on_fit=spec.fit_transform_on_fit,
                ))
            generator.operation_specs = tuple(new_specs)

        return generator

    def _build_classifier(self, classifier_name: str, classifier_params: dict[str, Any]):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import RidgeClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier

        classifier_name = classifier_name.lower()
        if classifier_name == 'logistic_regression':
            params = {'max_iter': 5000, 'random_state': 42}
            params.update(classifier_params)
            return LogisticRegression(**params)
        if classifier_name == 'random_forest':
            params = {'random_state': 42, 'n_estimators': 200}
            params.update(classifier_params)
            return RandomForestClassifier(**params)
        if classifier_name == 'svc':
            params = {'random_state': 42}
            params.update(classifier_params)
            return SVC(**params)
        if classifier_name == 'knn':
            params = {'n_neighbors': 5}
            params.update(classifier_params)
            return KNeighborsClassifier(**params)
        if classifier_name == 'ridge':
            params = {'random_state': 42}
            params.update(classifier_params)
            return RidgeClassifier(**params)
        if classifier_name == 'gradient_boosting':
            params = {'random_state': 42, 'max_iter': 200}
            params.update(classifier_params)
            return HistGradientBoostingClassifier(**params)
        if classifier_name == 'mlp':
            params = {'random_state': 42, 'max_iter': 1000, 'early_stopping': True}
            params.update(classifier_params)
            return MLPClassifier(**params)

        raise BenchmarkClassificationError(f'Unsupported sklearn classifier adapter: {classifier_name}')


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
    if name == 'sklearn_classifier':
        return KernelFeatureGeneratorClassifierAdapter(
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'feature_generator'),
            optional=spec.optional,
            params=dict(spec.params),
        )
    if name == 'fedot_industrial_classifier':
        return OptionalExternalClassifier(
            dependency_name='fedot',
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'external'),
        )
    raise BenchmarkClassificationError(f'Unsupported classification model adapter: {spec.adapter_name}')


__all__ = [
    "KernelEnsembleClassifierAdapter",
    "KernelFeatureGeneratorClassifierAdapter",
    "MajorityClassClassifier",
    "NearestCentroidClassifier",
    "OptionalExternalClassifier",
    "build_classification_model",
]

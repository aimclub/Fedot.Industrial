from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from fedot_ind.core.kernel_learning.contracts import FeatureBundle

DEFAULT_GENERATOR_NAMES = (
    "quantile_extractor",
    "wavelet_extractor",
    "fourier_extractor",
    "eigen_extractor",
)


def normalize_feature_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_time_series_tensor(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("X must contain at least one sample.")
    if array.ndim == 1:
        return array.reshape(1, 1, -1)
    if array.ndim == 2:
        return array.reshape(array.shape[0], 1, array.shape[1])
    if array.ndim == 3:
        return array
    return array.reshape(array.shape[0], int(np.prod(array.shape[1:-1])), array.shape[-1])


def to_fedot_input_data(
        X: Any,
        y: Any | None = None,
        *,
        task_type: str = "classification",
) -> InputData:
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    features = normalize_time_series_tensor(X)
    task_map = {
        "classification": Task(TaskTypesEnum.classification),
        "regression": Task(TaskTypesEnum.regression),
    }
    if task_type not in task_map:
        raise ValueError(f"Unsupported supervised task_type: {task_type}")
    target = None if y is None else np.asarray(y).reshape(-1, 1)
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=task_map[task_type],
        data_type=DataTypesEnum.image,
    )


def _unwrap_pipeline_output(value: Any) -> np.ndarray:
    if hasattr(value, "predict"):
        return normalize_feature_matrix(value.predict)
    return normalize_feature_matrix(value)


@dataclass
class PipelineFeatureGeneratorAdapter:
    name: str
    pipeline_factory: Any
    task_type_: str | None = None
    pipeline_: Any | None = None
    train_features_: np.ndarray | None = None

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        self.task_type_ = task_type
        self.pipeline_ = deepcopy(self.pipeline_factory).build()
        input_data = to_fedot_input_data(X, y, task_type=task_type)
        fit_result = self.pipeline_.fit(input_data)
        self.train_features_ = _unwrap_pipeline_output(fit_result)
        return self

    def transform(self, X: Any) -> FeatureBundle:
        if self.pipeline_ is None:
            raise ValueError(f"Feature generator {self.name!r} must be fitted before transform.")
        input_data = to_fedot_input_data(X, task_type=self.task_type_ or "classification")
        prediction = self.pipeline_.predict(input_data)
        features = _unwrap_pipeline_output(prediction)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "fedot_pipeline", "n_features": int(features.shape[1])},
        )

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        features = normalize_feature_matrix(self.train_features_)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "fedot_pipeline", "n_features": int(features.shape[1])},
        )


@dataclass
class IdentityFeatureGenerator:
    name: str = "identity"
    train_features_: np.ndarray | None = None

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        del y, task_type
        self.train_features_ = normalize_feature_matrix(X)
        return self

    def transform(self, X: Any) -> FeatureBundle:
        features = normalize_feature_matrix(X)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "identity", "n_features": int(features.shape[1])},
        )

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)


@dataclass
class SummaryFeatureGenerator:
    name: str = "statistical_summary"
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    train_features_: np.ndarray | None = None

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        del y, task_type
        self.train_features_ = self._build_features(X)
        return self

    def transform(self, X: Any) -> FeatureBundle:
        features = self._build_features(X)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "summary", "n_features": int(features.shape[1])},
        )

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return FeatureBundle(
            name=self.name,
            features=normalize_feature_matrix(self.train_features_),
            diagnostics={"source": "summary", "n_features": int(self.train_features_.shape[1])},
        )

    def _build_features(self, X: Any) -> np.ndarray:
        tensor = normalize_time_series_tensor(X)
        flat = tensor.reshape(tensor.shape[0], -1)
        q_values = np.quantile(flat, self.quantiles, axis=1).T
        start = flat[:, 0]
        end = flat[:, -1]
        slope = (end - start).reshape(-1, 1)
        features = np.hstack(
            [
                np.mean(flat, axis=1, keepdims=True),
                np.std(flat, axis=1, keepdims=True),
                np.min(flat, axis=1, keepdims=True),
                np.max(flat, axis=1, keepdims=True),
                q_values,
                slope,
            ]
        )
        return normalize_feature_matrix(features)


def build_generator_registry() -> dict[str, Callable[[], Any]]:
    registry: dict[str, Callable[[], Any]] = {
        "identity": IdentityFeatureGenerator,
        "statistical_summary": SummaryFeatureGenerator,
    }
    try:
        from fedot_ind.core.repository.constanst_repository import KERNEL_BASELINE_FEATURE_GENERATORS

        for name, pipeline_factory in KERNEL_BASELINE_FEATURE_GENERATORS.items():
            registry[name] = (
                lambda generator_name=name, factory=pipeline_factory: PipelineFeatureGeneratorAdapter(
                    name=generator_name,
                    pipeline_factory=factory,
                )
            )
    except Exception:
        # Keep pure NumPy generators importable in lightweight environments.
        pass
    return registry


def create_feature_generator(
        name: str,
        *,
        registry: dict[str, Callable[[], Any]] | None = None,
):
    source = registry or build_generator_registry()
    if name not in source:
        raise ValueError(f"Unknown kernel feature generator: {name}")
    return source[name]()

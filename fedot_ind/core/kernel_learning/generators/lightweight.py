from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fedot_ind.core.kernel_learning.contracts import FeatureBundle, FeatureInput, TargetInput
from fedot_ind.core.kernel_learning.generators.base import (
    KernelFeatureGeneratorMixin,
    normalize_feature_matrix,
    normalize_time_series_tensor,
)


@dataclass
class IdentityFeatureGenerator(KernelFeatureGeneratorMixin):
    name: str = "identity"
    train_features_: np.ndarray | None = None

    def fit(self, X: FeatureInput, y: TargetInput | None = None, *, task_type: str = "classification"):
        del y, task_type
        self.train_features_ = normalize_feature_matrix(X)
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        features = normalize_feature_matrix(X)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "identity", "n_features": int(features.shape[1])},
        )

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: str = "classification",
    ) -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)


@dataclass
class ShapeletFeatureGenerator(KernelFeatureGeneratorMixin):
    name: str = "shapelet_extractor"
    n_shapelets: int = 8
    window_size: int | None = None
    random_state: int = 42
    shapelets_: tuple[np.ndarray, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self):
        if self.n_shapelets < 1:
            raise ValueError("n_shapelets must be at least 1.")
        if self.window_size is not None and self.window_size < 1:
            raise ValueError("window_size must be at least 1.")

    def fit(self, X: FeatureInput, y: TargetInput | None = None, *, task_type: str = "classification"):
        del y, task_type
        tensor = normalize_time_series_tensor(X)
        n_samples, _, n_timestamps = tensor.shape
        window = self._resolve_window_size(n_timestamps)
        positions = _even_positions(n_timestamps - window + 1, self.n_shapelets)
        sample_indices = _even_positions(n_samples, self.n_shapelets)
        shapelets = []
        for shapelet_idx in range(self.n_shapelets):
            sample = tensor[sample_indices[shapelet_idx % len(sample_indices)]]
            position = positions[shapelet_idx % len(positions)]
            shapelets.append(sample[:, position:position + window].copy())
        self.shapelets_ = tuple(shapelets)
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        if not self.shapelets_:
            raise ValueError(f"Feature generator {self.name!r} must be fitted before transform.")
        tensor = normalize_time_series_tensor(X)
        features = np.column_stack([
            _min_shapelet_distance(tensor, shapelet)
            for shapelet in self.shapelets_
        ])
        return FeatureBundle(
            name=self.name,
            features=normalize_feature_matrix(features),
            diagnostics={
                "source": "shapelet",
                "n_shapelets": len(self.shapelets_),
                "window_size": int(self.shapelets_[0].shape[-1]),
                "n_features": len(self.shapelets_),
            },
        )

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: str = "classification",
    ) -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)

    def _resolve_window_size(self, n_timestamps: int) -> int:
        if self.window_size is not None:
            return min(int(self.window_size), n_timestamps)
        return max(1, min(n_timestamps, n_timestamps // 4 or 1))


@dataclass
class RandomProjectionEmbeddingFeatureGenerator(KernelFeatureGeneratorMixin):
    name: str = "embedding_extractor"
    n_components: int = 16
    random_state: int = 42
    source: str = "random_projection_embedding"
    components_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")

    def fit(self, X: FeatureInput, y: TargetInput | None = None, *, task_type: str = "classification"):
        del y, task_type
        features = normalize_feature_matrix(X)
        rng = np.random.default_rng(self.random_state)
        self.components_ = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(1, features.shape[1])),
            size=(features.shape[1], self.n_components),
        )
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        if self.components_ is None:
            raise ValueError(f"Feature generator {self.name!r} must be fitted before transform.")
        features = normalize_feature_matrix(X)
        if features.shape[1] != self.components_.shape[0]:
            raise ValueError("Embedding transform feature dimensionality must match fitted data.")
        embedding = np.tanh(features @ self.components_)
        return FeatureBundle(
            name=self.name,
            features=embedding,
            diagnostics={
                "source": self.source,
                "n_components": int(self.n_components),
                "random_state": int(self.random_state),
                "n_features": int(embedding.shape[1]),
            },
        )

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: str = "classification",
    ) -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)


def even_positions(size: int, count: int) -> np.ndarray:
    if size <= 0:
        return np.array([0], dtype=int)
    if count <= 1:
        return np.array([0], dtype=int)
    return np.unique(np.linspace(0, size - 1, num=count, dtype=int))


def min_shapelet_distance(tensor: np.ndarray, shapelet: np.ndarray) -> np.ndarray:
    n_samples, _, n_timestamps = tensor.shape
    window = shapelet.shape[-1]
    if n_timestamps < window:
        padded = np.pad(tensor, ((0, 0), (0, 0), (0, window - n_timestamps)), mode="edge")
        return np.linalg.norm((padded - shapelet).reshape(n_samples, -1), axis=1)
    distances = []
    for position in range(n_timestamps - window + 1):
        segment = tensor[:, :, position:position + window]
        distances.append(np.linalg.norm((segment - shapelet).reshape(n_samples, -1), axis=1))
    return np.min(np.vstack(distances), axis=0)


_even_positions = even_positions
_min_shapelet_distance = min_shapelet_distance


__all__ = [
    "IdentityFeatureGenerator",
    "RandomProjectionEmbeddingFeatureGenerator",
    "ShapeletFeatureGenerator",
    "even_positions",
    "min_shapelet_distance",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NystromApproximationPolicy:
    n_components: int = 32
    landmark_strategy: str = "even"
    ridge: float = 1e-10

    def __post_init__(self):
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative.")
        if self.landmark_strategy not in {"even", "first"}:
            raise ValueError(f"Unsupported landmark_strategy: {self.landmark_strategy}")


@dataclass
class NystromKernelApproximator:
    policy: NystromApproximationPolicy

    def fit(self, features: np.ndarray, kernel_fn: Any):
        matrix = np.asarray(features, dtype=float)
        n_samples = matrix.shape[0]
        n_landmarks = min(int(self.policy.n_components), n_samples)
        self.kernel_fn_ = kernel_fn
        self.landmark_indices_ = _select_landmarks(
            n_samples,
            n_landmarks,
            strategy=self.policy.landmark_strategy,
        )
        self.landmark_features_ = matrix[self.landmark_indices_]
        self.train_cross_ = kernel_fn(matrix, self.landmark_features_)
        landmark_kernel = kernel_fn(self.landmark_features_, self.landmark_features_)
        ridge = self.policy.ridge * np.eye(landmark_kernel.shape[0])
        self.landmark_pinv_ = np.linalg.pinv(landmark_kernel + ridge)
        return self

    def transform_train(self) -> np.ndarray:
        return self.train_cross_ @ self.landmark_pinv_ @ self.train_cross_.T

    def transform_cross(self, features: np.ndarray) -> np.ndarray:
        cross_to_landmarks = np.asarray(features, dtype=float)
        if cross_to_landmarks.shape[1] != self.landmark_features_.shape[1]:
            raise ValueError("Nystrom transform features must match fitted feature dimensionality.")
        test_cross = self.kernel_fn_(cross_to_landmarks, self.landmark_features_)
        return test_cross @ self.landmark_pinv_ @ self.train_cross_.T

    def diagnostics(self) -> dict[str, Any]:
        return {
            "approximation": "nystrom",
            "n_components": int(len(self.landmark_indices_)),
            "landmark_strategy": self.policy.landmark_strategy,
            "ridge": float(self.policy.ridge),
            "landmark_indices": tuple(int(index) for index in self.landmark_indices_),
        }


def _select_landmarks(n_samples: int, n_landmarks: int, *, strategy: str) -> np.ndarray:
    if n_samples < 1:
        raise ValueError("At least one sample is required for Nystrom approximation.")
    if strategy == "first":
        return np.arange(n_landmarks, dtype=int)
    if n_landmarks == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n_samples - 1, num=n_landmarks, dtype=int)

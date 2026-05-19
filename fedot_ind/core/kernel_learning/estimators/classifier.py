from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.svm import SVC

from .base import KernelEnsembleBase


class KernelEnsembleClassifier(KernelEnsembleBase):
    task_type = "classification"

    def __init__(
            self,
            generator_names: tuple[str, ...] | list[str] | None = None,
            kernel: str = "rbf",
            gamma: str | float = "scale",
            normalize: str | None = "trace",
            center: bool = False,
            psd_correction: str | None = "clip",
            psd_tol: float = 1e-8,
            complexity_penalty: float = 0.01,
            redundancy_penalty: float = 0.05,
            min_weight: float = 0.05,
            target_gamma: str | float = "scale",
            importance_threshold: float = 0.05,
            importance_fallback_top_n: int = 1,
            importance_max_union_size: int = 3,
            C: float = 1.0,
            probability: bool = True,
            random_state: int = 42,
            head: Any | None = None,
            torch_device: Any = "auto",
    ):
        super().__init__(
            generator_names=generator_names,
            kernel=kernel,
            gamma=gamma,
            normalize=normalize,
            center=center,
            psd_correction=psd_correction,
            psd_tol=psd_tol,
            complexity_penalty=complexity_penalty,
            redundancy_penalty=redundancy_penalty,
            min_weight=min_weight,
            target_gamma=target_gamma,
            importance_threshold=importance_threshold,
            importance_fallback_top_n=importance_fallback_top_n,
            importance_max_union_size=importance_max_union_size,
            torch_device=torch_device,
        )
        self.C = C
        self.probability = probability
        self.random_state = random_state
        self.head = head
        self.torch_device = torch_device

    def fit(self, X: Any, y: Any):
        y_array = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y_array)
        train_kernel = self._fit_kernel_layer(X, y_array)
        self.head_ = self._clone_head(self.head) if self.head is not None else SVC(
            kernel="precomputed",
            probability=self.probability,
            C=self.C,
            random_state=self.random_state,
        )
        self.head_.fit(train_kernel, y_array)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.head_.predict(self._combine_test_kernels(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        test_kernel = self._combine_test_kernels(X)
        if hasattr(self.head_, "predict_proba"):
            return self.head_.predict_proba(test_kernel)
        predictions = self.head_.predict(test_kernel)
        probabilities = np.zeros((len(predictions), len(self.classes_)), dtype=float)
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        for row, label in enumerate(predictions):
            probabilities[row, class_to_index[label]] = 1.0
        return probabilities

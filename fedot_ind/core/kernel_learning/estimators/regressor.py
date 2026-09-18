from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

from fedot_ind.core.kernel_learning.contracts import (
    FeatureInput,
    KernelConfigValidationError,
    KernelTaskType,
    TargetInput,
)

from .base import KernelEnsembleBase, collect_kernel_base_params


class KernelEnsembleRegressor(KernelEnsembleBase):
    task_type = KernelTaskType.REGRESSION

    def __init__(
            self,
            generator_names: tuple[str, ...] | list[str] | None = None,
            kernel: str = "rbf",
            gamma: str | float = "scale",
            normalize: str | None = "trace",
            center: bool = False,
            psd_correction: str | None = "clip",
            psd_tol: float = 1e-8,
            kernel_approximation: str | None = None,
            nystrom_components: int | None = None,
            complexity_penalty: float = 0.01,
            redundancy_penalty: float = 0.05,
            min_weight: float = 0.05,
            target_gamma: str | float = "scale",
            selector_optimizer: str = "projected_gradient",
            selector_max_iter: int = 100,
            selector_tol: float = 1e-6,
            selector_step_size: float = 0.2,
            importance_threshold: float = 0.05,
            importance_fallback_top_n: int = 1,
            importance_max_union_size: int = 3,
            alpha: float = 1.0,
            head_type: str = "kernel_ridge",
            C: float = 1.0,
            epsilon: float = 0.1,
            head: Any | None = None,
            torch_device: Any = "auto",
            kernel_cache_enabled: bool = True,
            kernel_cache_namespace: str = "kernel_ensemble",
    ):
        super().__init__(**collect_kernel_base_params(locals()))
        self.alpha = alpha
        self.head_type = head_type
        self.C = C
        self.epsilon = epsilon
        self.head = head

    def fit(self, X: FeatureInput, y: TargetInput):
        y_array = np.asarray(y, dtype=float).reshape(-1)
        train_kernel = self._fit_kernel_layer(X, y_array)
        self.head_ = self._clone_head(self.head) if self.head is not None else self._build_default_head()
        self.head_.fit(train_kernel, y_array)
        return self

    def predict(self, X: FeatureInput) -> np.ndarray:
        return np.asarray(self.head_.predict(self._combine_test_kernels(X)), dtype=float).reshape(-1)

    def _build_default_head(self):
        normalized = self.head_type.lower()
        if normalized == "kernel_ridge":
            return KernelRidge(kernel="precomputed", alpha=self.alpha)
        if normalized == "svr":
            return SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
        raise KernelConfigValidationError(f"Unsupported regression head_type: {self.head_type}")

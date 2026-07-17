from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from fedot_ind.core.kernel_learning.contracts import (
    FeatureInput,
    KernelConfigValidationError,
    KernelTaskType,
    TargetInput,
)

from .base import KernelEnsembleBase, collect_kernel_base_params


@dataclass
class OKHSForecastHeadAdapter:
    alpha: float = 1.0
    diagnostics_: dict[str, Any] = field(default_factory=dict, init=False)

    def fit(self, kernel: np.ndarray, y: np.ndarray):
        self.model_ = KernelRidge(kernel="precomputed", alpha=self.alpha)
        self.model_.fit(kernel, y)
        self.diagnostics_ = {
            "head_type": "okhs_kernel_ridge_adapter",
            "alpha": float(self.alpha),
            "target_shape": tuple(int(size) for size in np.asarray(y).shape),
        }
        return self

    def predict(self, kernel: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(kernel), dtype=float)


class KernelEnsembleForecaster(KernelEnsembleBase):
    task_type = KernelTaskType.FORECASTING

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
            forecast_horizon: int | None = None,
            head_type: str = "kernel_ridge",
            alpha: float = 1.0,
            head: Any | None = None,
            torch_device: Any = "auto",
            kernel_cache_enabled: bool = True,
            kernel_cache_namespace: str = "kernel_ensemble",
    ):
        super().__init__(**collect_kernel_base_params(locals()))
        self.forecast_horizon = forecast_horizon
        self.head_type = head_type
        self.alpha = alpha
        self.head = head

    def fit(self, X: FeatureInput, y: TargetInput):
        y_array = _normalize_forecast_target(y, forecast_horizon=self.forecast_horizon)
        train_kernel = self._fit_kernel_layer(X, y_array)
        self.forecast_horizon_ = int(y_array.shape[1]) if y_array.ndim == 2 else 1
        self.head_ = self._clone_head(self.head) if self.head is not None else self._build_default_head()
        self.head_.fit(train_kernel, y_array)
        self.head_diagnostics_ = getattr(self.head_, "diagnostics_", {"head_type": self.head_type})
        return self

    def predict(self, X: FeatureInput) -> np.ndarray:
        prediction = np.asarray(self.head_.predict(self._combine_test_kernels(X)), dtype=float)
        if self.forecast_horizon_ == 1:
            return prediction.reshape(-1)
        return prediction.reshape(prediction.shape[0], self.forecast_horizon_)

    def _build_default_head(self):
        normalized = self.head_type.lower()
        if normalized == "kernel_ridge":
            return KernelRidge(kernel="precomputed", alpha=self.alpha)
        if normalized in {"okhs", "okhs_fdmd", "okhs_kernel_ridge"}:
            return OKHSForecastHeadAdapter(alpha=self.alpha)
        raise KernelConfigValidationError(f"Unsupported forecasting head_type: {self.head_type}")


def _normalize_forecast_target(y: Any, *, forecast_horizon: int | None) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if forecast_horizon is not None:
        if forecast_horizon < 1:
            raise KernelConfigValidationError("forecast_horizon must be at least 1.")
        if values.shape[1] != forecast_horizon:
            raise KernelConfigValidationError(
                f"forecast_horizon={forecast_horizon} does not match target width {values.shape[1]}."
            )
    return values

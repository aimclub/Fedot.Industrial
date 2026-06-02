from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForecastTargetSpec:
    horizon_weights: tuple[float, ...] | None = None
    forecastability_prior: float = 0.0

    def __post_init__(self):
        if self.forecastability_prior < 0.0:
            raise ValueError("forecastability_prior must be non-negative.")
        if self.horizon_weights is not None:
            if not self.horizon_weights:
                raise ValueError("horizon_weights must not be empty.")
            if any(weight < 0.0 for weight in self.horizon_weights):
                raise ValueError("horizon_weights must be non-negative.")
            if sum(self.horizon_weights) <= 0.0:
                raise ValueError("At least one horizon weight must be positive.")


@dataclass
class TargetKernelBuilder:
    task_type: str
    gamma: str | float = "scale"
    forecast_spec: ForecastTargetSpec | None = None

    def build(self, y: Any) -> np.ndarray:
        task_type = _canonical_task_type(self.task_type)
        if task_type == "classification":
            labels = np.asarray(y).reshape(-1)
            return (labels[:, None] == labels[None, :]).astype(float)
        if task_type == "regression":
            values = np.asarray(y, dtype=float).reshape(-1, 1)
            gamma = self._resolve_gamma(values)
            diff = values - values.T
            return np.exp(-gamma * diff ** 2)
        if task_type == "forecasting":
            return self._build_forecasting_kernel(y)
        raise ValueError(f"Unsupported target kernel task_type: {self.task_type}")

    def _build_forecasting_kernel(self, y: Any) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        spec = self.forecast_spec or ForecastTargetSpec()
        horizon_weights = _resolve_horizon_weights(values.shape[1], spec.horizon_weights)
        kernel = np.zeros((values.shape[0], values.shape[0]), dtype=float)
        for horizon_idx, horizon_weight in enumerate(horizon_weights):
            horizon_values = values[:, horizon_idx].reshape(-1, 1)
            gamma = self._resolve_gamma(horizon_values)
            diff = horizon_values - horizon_values.T
            kernel += horizon_weight * np.exp(-gamma * diff ** 2)
        if spec.forecastability_prior > 0.0:
            trend = _forecastability_similarity(values)
            kernel = (1.0 - spec.forecastability_prior) * kernel + spec.forecastability_prior * trend
        return kernel

    def _resolve_gamma(self, values: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        variance = float(np.var(values))
        if str(self.gamma).lower() == "scale":
            return 1.0 / variance if variance > 1e-12 else 1.0
        if str(self.gamma).lower() == "auto":
            return 1.0
        raise ValueError(f"Unsupported target gamma: {self.gamma}")


def _canonical_task_type(task_type: str) -> str:
    normalized = str(task_type).lower()
    if normalized in {"classification", "regression", "forecasting"}:
        return normalized
    if normalized == "ts_forecasting":
        return "forecasting"
    raise ValueError(f"Unsupported target kernel task_type: {task_type}")


def _resolve_horizon_weights(n_horizons: int, weights: tuple[float, ...] | None) -> np.ndarray:
    if weights is None:
        return np.ones(n_horizons, dtype=float) / n_horizons
    if len(weights) != n_horizons:
        raise ValueError(f"horizon_weights length {len(weights)} does not match target horizons {n_horizons}.")
    values = np.asarray(weights, dtype=float)
    return values / np.sum(values)


def _forecastability_similarity(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    normalized = centered / norms
    return np.clip(normalized @ normalized.T, a_min=0.0, a_max=1.0)

"""Unified metrics API: one dataset per call, dict results, parameterized metric specs.

See ``README.md`` in this package. Legacy pandas DataFrame output remains in
``metrics_implementation.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fedot_ind.core.metrics._exceptions import (
    MetricError,
    MetricNotFoundError,
    MetricValidationError,
)
from fedot_ind.core.metrics.metric_library import get_metric

__all__ = [
    'MetricError',
    'MetricNotFoundError',
    'MetricValidationError',
    'QualityMetric',
    'ClassificationMetric',
    'AnomalyMetric',
    'RegressionMetric',
    'ForecastingMetric',
    'calculate_classification_metric',
    'calculate_detection_metric',
    'calculate_regression_metric',
    'calculate_forecasting_metric',
]


# ---------------------------------------------------------------------------
# Input helpers (no interval support)
# ---------------------------------------------------------------------------

def _as_array(values: Any, *, dtype=None) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _reject_nested_series(values: Any) -> None:
    """Reject batched multi-series input like ``[[0, 1], [1, 0]]``."""
    if isinstance(values, np.ndarray):
        return
    if isinstance(values, (list, tuple)) and len(values) > 0 and isinstance(values[0], (list, tuple)):
        inner = values[0]
        if len(inner) > 0 and not isinstance(inner[0], (list, tuple)):
            raise MetricValidationError(
                'multiple series in one call are not supported; evaluate one dataset at a time.',
            )


def _labels(target_or_labels: Any) -> np.ndarray:
    _reject_nested_series(target_or_labels)
    array = _as_array(target_or_labels)
    if array.ndim >= 2 and array.shape[-1] > 1:
        array = np.argmax(array, axis=-1)
    return array.reshape(-1)


def _float_pair(target: Any, predicted: Any) -> tuple[np.ndarray, np.ndarray]:
    y_true = _as_array(target, dtype=float).reshape(-1)
    y_pred = _as_array(predicted, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred):
        raise MetricValidationError(
            f'target and predicted must have the same length ({len(y_true)} != {len(y_pred)}).',
        )
    return y_true, y_pred


def _probs(probs: Any | None, *, n_samples: int) -> np.ndarray | None:
    if probs is None:
        return None
    array = _as_array(probs, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1)
    if array.ndim != 2 or array.shape[0] != n_samples:
        raise MetricValidationError('predicted_probs must be 1D or 2D with shape (n_samples, n_classes).')
    return array


def _normalize_metrics(
        metrics: Sequence[str | Mapping[str, Any]] | str | None,
        default: tuple[str, ...],
) -> tuple[str | Mapping[str, Any], ...]:
    if metrics is None:
        return default
    if isinstance(metrics, str):
        return (metrics,)
    return tuple(metrics)


def _parse_spec(spec: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping):
        name = spec.get('name')
        if not name:
            raise MetricValidationError('Metric spec dict must include "name".')
        params = spec.get('params') or {}
        if not isinstance(params, Mapping):
            raise MetricValidationError('Metric spec "params" must be a mapping.')
        return str(name), dict(params)
    raise MetricValidationError(f'Invalid metric spec: {type(spec).__name__}.')


def _result_keys(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    keys: list[str] = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            keys.append(name)
        else:
            counts[name] += 1
            keys.append(f'{name}__{counts[name]}')
    return keys


# ---------------------------------------------------------------------------
# QualityMetric and task evaluators
# ---------------------------------------------------------------------------

class QualityMetric:
    """Base evaluator: parses metric specs and dispatches to ``metric_library``."""

    task: str = 'NONE'

    def __init__(self, rounding_order: int = 4) -> None:
        self.rounding_order = int(rounding_order)
        self.target: np.ndarray = np.array([])
        self.predicted: np.ndarray = np.array([])

    def _extra_kwargs(self, _metric_name: str) -> dict[str, Any]:
        return {}

    def metric_evaluation(
            self,
            metrics_specs: Sequence[str | Mapping[str, Any]],
    ) -> dict[str, float | list | dict]:
        if not metrics_specs:
            return {}
        parsed = [_parse_spec(s) for s in metrics_specs]
        names = [n for n, _ in parsed]
        keys = _result_keys(names)
        out: dict[str, float | list | dict] = {}
        for key, (name, params) in zip(keys, parsed):
            fn = get_metric(self.task, name)
            value = fn(self.target, self.predicted, **{**self._extra_kwargs(name), **params})
            if isinstance(value, (float, np.floating)):
                out[key] = round(float(value), self.rounding_order)
            else:
                out[key] = value
        return out


class ClassificationMetric(QualityMetric):
    """Classification metrics on label sequences (optional probabilities)."""

    task = 'classification'

    def __init__(
            self,
            target: Any,
            predicted_labels: Any,
            predicted_probs: Any | None = None,
            rounding_order: int = 4,
            default_value: float = 0.0,
    ) -> None:
        super().__init__(rounding_order=rounding_order)
        self.target = _labels(target)
        self.predicted = _labels(predicted_labels)
        if len(self.target) != len(self.predicted):
            raise MetricValidationError('target and predicted_labels length mismatch.')
        self._probs = _probs(predicted_probs, n_samples=len(self.target))
        self._default_value = default_value

    def _extra_kwargs(self, _metric_name: str) -> dict[str, Any]:
        return {'predicted_probs': self._probs, 'default_value': self._default_value}


class RegressionMetric(QualityMetric):
    """Regression metrics on float vectors."""

    task = 'regression'

    def __init__(self, target: Any, predicted: Any, rounding_order: int = 4) -> None:
        super().__init__(rounding_order=rounding_order)
        self.target, self.predicted = _float_pair(target, predicted)


class ForecastingMetric(QualityMetric):
    """Forecasting horizon metrics (aggregate or pointwise via metric params)."""

    task = 'forecasting'

    def __init__(
            self,
            target: Any,
            predicted: Any,
            rounding_order: int = 4,
            train_data: Any | None = None,
            seasonality: int = 1,
    ) -> None:
        super().__init__(rounding_order=rounding_order)
        self.target, self.predicted = _float_pair(target, predicted)
        self._train = _as_array(train_data if train_data is not None else target, dtype=float).reshape(-1)
        self._seasonality = int(seasonality)

    def _extra_kwargs(self, _metric_name: str) -> dict[str, Any]:
        return {'y_train': self._train, 'seasonal_period': self._seasonality}


class AnomalyMetric(QualityMetric):
    """Anomaly detection on binary label sequences (0/1)."""

    task = 'detection'

    def __init__(
            self,
            target: Any,
            predicted_labels: Any,
            rounding_order: int = 4,
            **detection_params: Any,
    ) -> None:
        super().__init__(rounding_order=rounding_order)
        self.target = _labels(target)
        self.predicted = _labels(predicted_labels)
        if len(self.target) != len(self.predicted):
            raise MetricValidationError('target and predicted_labels length mismatch.')
        self._detection_params = detection_params

    def _extra_kwargs(self, _metric_name: str) -> dict[str, Any]:
        return dict(self._detection_params)


# ---------------------------------------------------------------------------
# Public calculate_* API
# ---------------------------------------------------------------------------

def calculate_classification_metric(
        target: Any,
        predicted_labels: Any,
        predicted_probs: Any | None = None,
        metrics: Sequence[str | Mapping[str, Any]] | str | None = None,
        rounding_order: int = 4,
        **kwargs: Any,
) -> dict[str, float | list | dict]:
    """Compute classification metrics for one dataset."""
    specs = _normalize_metrics(metrics, ('f1','accuracy'))
    return ClassificationMetric(target, 
                                predicted_labels, 
                                predicted_probs,
                                rounding_order=rounding_order,
                                default_value=float(kwargs.get('default_value', 0.0))
                                ).metric_evaluation(specs)


def calculate_regression_metric(
        target: Any,
        predicted: Any,
        metrics: Sequence[str | Mapping[str, Any]] | str | None = None,
        rounding_order: int = 4,
        **kwargs: Any,
) -> dict[str, float | list | dict]:
    """Compute regression metrics for one dataset."""
    del kwargs
    specs = _normalize_metrics(metrics, ('r2', 'rmse', 'mae'))
    return RegressionMetric(target, 
                            predicted, 
                            rounding_order=rounding_order
                            ).metric_evaluation(specs)


def calculate_forecasting_metric(
        target: Any,
        predicted: Any,
        metrics: Sequence[str | Mapping[str, Any]] | str | None = None,
        rounding_order: int = 4,
        **kwargs: Any,
) -> dict[str, float | list | dict]:
    """Compute forecasting metrics for one horizon."""
    specs = _normalize_metrics(metrics, ('smape', 'rmse', 'mape'))
    return ForecastingMetric(
        target, predicted,
        rounding_order=rounding_order,
        train_data=kwargs.get('train_data'),
        seasonality=int(kwargs.get('seasonality', kwargs.get('seasonal_period', 1))),
    ).metric_evaluation(specs)


def calculate_detection_metric(
        target: Any,
        predicted_labels: Any,
        metrics: Sequence[str | Mapping[str, Any]] | str | None = None,
        rounding_order: int = 4,
        **kwargs: Any,
) -> dict[str, float | list | dict]:
    """Compute anomaly detection metrics on binary label sequences."""
    specs = _normalize_metrics(metrics, ('f1','accuracy'))
    return AnomalyMetric(
        target, predicted_labels,
        rounding_order=rounding_order,
        **kwargs,
    ).metric_evaluation(specs)

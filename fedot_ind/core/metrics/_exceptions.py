"""Exceptions raised by the metrics package."""


class MetricError(Exception):
    """Base exception for all metric-related failures."""


class MetricValidationError(MetricError):
    """Raised when inputs or metric parameters are invalid.

    Examples: length mismatch, missing ``predicted_probs`` for ``roc_auc``,
    multi-series target in a single-dataset API call.
    """


class MetricNotFoundError(MetricError):
    """Raised when a requested metric name is not registered for the task."""

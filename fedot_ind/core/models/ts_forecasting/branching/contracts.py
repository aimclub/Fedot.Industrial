"""Forecast branches exchange windows or forecasts, never an ambiguous tensor."""

from enum import Enum


class ForecastMergeKind(str, Enum):
    FEATURES = 'forecast_feature_merge'
    PREDICTIONS = 'forecast_prediction_merge'


class ForecastBranchError(ValueError):
    """An explicitly incompatible branch cannot be silently truncated or aligned."""


def merge_kind(tags):
    kinds = tuple(kind for kind in ForecastMergeKind if kind.value in tags)
    if len(kinds) > 1:
        raise ForecastBranchError('A merge cannot consume both features and predictions')
    return kinds[0] if kinds else None

# benchmark/v2/forecasting_result.py

from __future__ import annotations

from typing import Any, Iterable
import numpy as np

from .core import ForecastResult, ForecastingSeriesRecord, QuantilePredictionRecord, RunStatus


class ForecastResultValidationError(ValueError):
    """Raised when ForecastResult cannot be aligned with requested forecast horizon."""
    pass


def _as_1d_float_array(value: Any, *, field_name: str) -> np.ndarray:
    """Convert value to 1D float numpy array."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ForecastResultValidationError(f'ForecastResult.{field_name} is empty.')
    if not np.all(np.isfinite(array)):
        raise ForecastResultValidationError(f'ForecastResult.{field_name} contains non-finite values.')
    return array


def _validate_vector_length(array: np.ndarray, *, expected_length: int, field_name: str) -> np.ndarray:
    """Validate array length matches expected horizon."""
    if len(array) != expected_length:
        raise ForecastResultValidationError(
            f'ForecastResult.{field_name} has length {len(array)}, '
            f'expected horizon length {expected_length}.'
        )
    return array


def _normalize_quantile_level(level: Any) -> float:
    """Validate and normalize quantile level to float in (0,1)."""
    try:
        quantile = float(level)
    except (TypeError, ValueError) as exc:
        raise ForecastResultValidationError(f'Invalid quantile level: {level!r}.') from exc

    if not 0.0 < quantile < 1.0:
        raise ForecastResultValidationError(
            f'Quantile level must be inside (0, 1), got {quantile}.'
        )
    return quantile


def _normalize_coverage_level(level: Any) -> float:
    """Validate and normalize coverage level to float in (0,1)."""
    try:
        coverage = float(level)
    except (TypeError, ValueError) as exc:
        raise ForecastResultValidationError(f'Invalid interval coverage level: {level!r}') from exc

    if not 0.0 < coverage < 1.0:
        raise ForecastResultValidationError(
            f'Interval coverage must be inside (0, 1), got {coverage}.'
        )
    return coverage


def coerce_forecast_result(raw_result: Any) -> ForecastResult:
    """Convert legacy adapter outputs into ForecastResult.

    Supported inputs:
    - ForecastResult
    - (prediction, metadata)
    - bare prediction vector
    """
    if isinstance(raw_result, ForecastResult):
        return raw_result

    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        prediction, metadata = raw_result
        return ForecastResult(
            mean=np.asarray(prediction, dtype=float).reshape(-1),
            metadata=dict(metadata or {}),
        )

    # Bare prediction vector
    return ForecastResult(
        mean=np.asarray(raw_result, dtype=float).reshape(-1),
        metadata={},
    )


def describe_forecast_result_kind(result: ForecastResult) -> str:
    """Return a string describing which fields are present."""
    parts: list[str] = []
    if result.mean is not None:
        parts.append('mean')
    if result.quantiles:
        parts.append('quantiles')
    if result.intervals:
        parts.append('intervals')
    if result.samples is not None:
        parts.append('samples')
    if result.fitted_values is not None:
        parts.append('fitted_values')
    if result.residuals is not None:
        parts.append('residuals')
    return '+'.join(parts) if parts else 'empty'


def validate_forecast_result_shapes(result: ForecastResult, *, horizon: int) -> None:
    """Validate shapes of all horizon-aligned outputs."""

    if result.mean is not None:
        mean = _as_1d_float_array(result.mean, field_name='mean')
        _validate_vector_length(mean, expected_length=horizon, field_name='mean')

    for raw_level, values in dict(result.quantiles).items():
        quantile = _normalize_quantile_level(raw_level)
        array = _as_1d_float_array(values, field_name=f'quantiles[{quantile}]')
        _validate_vector_length(array, expected_length=horizon, field_name=f'quantiles[{quantile}]')

    for raw_coverage, bounds in dict(result.intervals).items():
        coverage = _normalize_coverage_level(raw_coverage)
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ForecastResultValidationError(
                f'ForecastResult.intervals[{coverage}] must be a pair: (lower, upper).'
            )
        lower, upper = bounds
        lower_array = _as_1d_float_array(lower, field_name=f'intervals[{coverage}].lower')
        upper_array = _as_1d_float_array(upper, field_name=f'intervals[{coverage}].upper')
        _validate_vector_length(lower_array, expected_length=horizon, field_name=f'intervals[{coverage}].lower')
        _validate_vector_length(upper_array, expected_length=horizon, field_name=f'intervals[{coverage}].upper')

    if result.samples is not None:
        samples = np.asarray(result.samples, dtype=float)
        if samples.ndim != 2:
            raise ForecastResultValidationError(
                'ForecastResult.samples must have shape [n_samples, horizon].'
            )
        if samples.shape[0] == 0:
            raise ForecastResultValidationError('ForecastResult.samples has zero samples.')
        if samples.shape[1] != horizon:
            raise ForecastResultValidationError(
                f'ForecastResult.samples has horizon length {samples.shape[1]}, '
                f'expected {horizon}.'
            )
        if not np.all(np.isfinite(samples)):
            raise ForecastResultValidationError('ForecastResult.samples contains non-finite values.')


def resolve_point_forecast(result: ForecastResult) -> tuple[np.ndarray, dict[str, Any]]:
    """Return point forecast vector used by point metrics.

    Fallback order:
    1. mean
    2. closest available quantile to q=0.5
    3. sample mean
    """
    metadata = dict(result.metadata)

    if result.mean is not None:
        return np.asarray(result.mean, dtype=float).reshape(-1), metadata

    if result.quantiles:
        normalized_quantiles = {
            _normalize_quantile_level(level): np.asarray(values, dtype=float).reshape(-1)
            for level, values in dict(result.quantiles).items()
        }
        median_key = min(normalized_quantiles, key=lambda level: abs(level - 0.5))
        metadata['point_forecast_fallback'] = f'quantile_{median_key:g}'
        return normalized_quantiles[median_key], metadata

    if result.samples is not None:
        samples = np.asarray(result.samples, dtype=float)
        if samples.ndim != 2:
            raise ForecastResultValidationError(
                'ForecastResult.samples must have shape [n_samples, horizon].'
            )
        metadata['point_forecast_fallback'] = 'sample_mean'
        return np.mean(samples, axis=0).reshape(-1), metadata

    raise ForecastResultValidationError(
        'ForecastResult does not contain mean, quantiles, or samples.'
    )


def iter_quantile_prediction_records(
        run_id: str,
        series_record: ForecastingSeriesRecord,
        model_name: str,
        actual: np.ndarray,
        forecast_result: ForecastResult,
) -> Iterable[QuantilePredictionRecord]:
    actual_vector = np.asarray(actual, dtype=float).reshape(-1)

    for raw_level, values in dict(forecast_result.quantiles).items():
        quantile = _normalize_quantile_level(raw_level)
        forecast_vector = np.asarray(values, dtype=float).reshape(-1)
        _validate_vector_length(
            forecast_vector,
            expected_length=len(actual_vector),
            field_name=f'quantiles[{quantile}]',
        )

        for horizon_index, (actual_value, forecast_value) in enumerate(
                zip(actual_vector, forecast_vector), start=1
        ):
            yield QuantilePredictionRecord(
                run_id=run_id,
                benchmark=series_record.benchmark,
                dataset_name=series_record.dataset_name,
                subset=series_record.subset,
                series_id=series_record.series_id,
                model_name=model_name,
                horizon_index=horizon_index,
                quantile=float(quantile),
                y_true=float(actual_value),
                y_pred=float(forecast_value),
                status=RunStatus.SUCCESS,
            )
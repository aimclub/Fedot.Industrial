"""Published M4 and Monash scoring rules, separate from legacy metrics."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..protocols.schema import ProtocolError, integer


# Horizons and seasonal periods of the competition groups.
M4_GROUPS = {'Yearly': (6, 1), 'Quarterly': (8, 4), 'Monthly': (18, 12),
             'Weekly': (13, 1), 'Daily': (14, 1), 'Hourly': (48, 24)}


class ForecastModelScope(str, Enum):
    LOCAL = 'local'
    PANEL = 'panel'
    MULTIVARIATE = 'multivariate'


def vector(values, label):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ProtocolError(f'{label} must be a non-empty finite vector')
    return result


def seasonality_lag(histories, declared_periods):
    """Monash §4.3: use the shortest cycle if every series reaches that length."""
    lengths = tuple(len(history) for history in histories)
    periods = tuple(integer(period, 'seasonal period') for period in declared_periods)
    if not lengths or not periods:
        raise ProtocolError('Monash requires histories and declared seasonal periods')
    lag = min(periods)
    return lag if min(lengths) >= lag else 1


def reference_mase(history, truth, prediction, lag):
    history, truth, prediction = (vector(values, label) for values, label in
                                  ((history, 'history'), (truth, 'truth'), (prediction, 'prediction')))
    integer(lag, 'scale lag')
    if len(history) <= lag or truth.shape != prediction.shape:
        raise ProtocolError('MASE needs aligned outputs and training longer than the scale lag')
    scale = np.abs(history[lag:] - history[:-lag]).mean()
    if scale == 0:
        raise ProtocolError('MASE scale is zero for the declared training history')
    return float(np.abs(truth - prediction).mean() / scale)


def reference_smape(truth, prediction):
    truth, prediction = vector(truth, 'truth'), vector(prediction, 'prediction')
    if truth.shape != prediction.shape:
        raise ProtocolError('sMAPE needs aligned truth and prediction')
    scale = np.abs(truth) + np.abs(prediction)
    ratios = np.divide(np.abs(truth - prediction), scale, out=np.zeros_like(scale), where=scale != 0)
    return float(200 * ratios.mean())


def _m4_seasonal(history, period):
    if period == 1 or len(history) < 3 * period:
        return False
    centered = history - history.mean()
    variance = np.dot(centered, centered)
    if variance == 0:
        return False
    acf = np.array([np.dot(centered[:-lag], centered[lag:]) / variance for lag in range(1, period + 1)])
    limit = 1.645 * np.sqrt((1 + 2 * np.dot(acf[:-1], acf[:-1])) / len(history))
    return abs(acf[-1]) > limit


def m4_naive2(history, horizon, period):
    """Classical multiplicative decomposition after the organizers' ACF test."""
    history = vector(history, 'Naive2 history')
    integer(horizon, 'horizon')
    integer(period, 'seasonal period')
    if not _m4_seasonal(history, period):
        return np.repeat(history[-1], horizon)
    from statsmodels.tsa.seasonal import seasonal_decompose

    if np.any(history <= 0):
        raise ProtocolError('Multiplicative Naive2 requires positive seasonal training values')
    seasonal = seasonal_decompose(history, model='multiplicative', period=period).seasonal
    return (history[-1] / seasonal[-1]) * np.resize(seasonal[-period:], horizon)


@dataclass(frozen=True)
class ForecastEvidence:
    series_id: str
    history: object
    truth: object
    prediction: object
    target_times: tuple[str, ...]
    prediction_times: tuple[str, ...]
    organizer_naive2: object | None = None


def _validated_rows(rows, horizon=None):
    rows = tuple(rows)
    if not rows or len({row.series_id for row in rows}) != len(rows):
        raise ProtocolError('The scoring group must contain unique series')
    for row in rows:
        if (row.target_times != row.prediction_times or len(row.target_times) != len(row.truth)
                or len(set(row.target_times)) != len(row.target_times)
                or (horizon is not None and len(row.truth) != horizon)):
            raise ProtocolError(f'Unaligned forecast dates or wrong horizon for {row.series_id}')
    return rows


def score_m4(group, rows):
    if group not in M4_GROUPS:
        raise ProtocolError(f'Unknown M4 frequency group: {group!r}')
    horizon, period = M4_GROUPS[group]
    rows = _validated_rows(rows, horizon)
    from_organizers = [row.organizer_naive2 is not None for row in rows]
    if any(from_organizers) and not all(from_organizers):
        raise ProtocolError('Do not mix imported and reproduced Naive2 forecasts')
    scores = []
    for row in rows:
        base = row.organizer_naive2 if all(from_organizers) else m4_naive2(row.history, horizon, period)
        scores.append([reference_smape(row.truth, row.prediction),
                       reference_mase(row.history, row.truth, row.prediction, period),
                       reference_smape(row.truth, base), reference_mase(row.history, row.truth, base, period)])
    smape, mase, base_smape, base_mase = np.asarray(scores).mean(axis=0)
    if base_smape == 0 or base_mase == 0:
        raise ProtocolError('OWA is undefined when an aggregate Naive2 error is zero')
    return {'protocol': 'm4@1', 'group': group, 'horizon': horizon, 'series_count': len(rows),
            'smape': float(smape), 'mase': float(mase), 'owa': float((smape / base_smape + mase / base_mase) / 2),
            'naive2_source': 'organizers' if all(from_organizers) else 'reproduced'}


def score_monash(rows, periods, *, scope: ForecastModelScope):
    rows = _validated_rows(rows)
    scope = ForecastModelScope(scope)
    lag = seasonality_lag((row.history for row in rows), periods)
    results = [reference_mase(row.history, row.truth, row.prediction, lag) for row in rows]
    return {'protocol': 'monash@1', 'scope': scope.value, 'lag': lag, 'series_count': len(rows),
            'mean_mase': float(np.mean(results)), 'median_mase': float(np.median(results))}


def collection_identity(benchmark, collection):
    """M4 subsets in Monash keep their source-group identity for coverage."""
    name = collection.lower().replace(' ', '_').removesuffix('_dataset')
    if benchmark == 'monash' and name.startswith('m4_') and name[3:].title() in M4_GROUPS:
        return 'm4', name[3:]
    return benchmark, name


def unique_forecast_collections(source_collections):
    """Prevent duplicate weighting across native and mirrored benchmark collections."""
    identities = {}
    for benchmark, collection in source_collections:
        identity = collection_identity(benchmark, collection)
        if identity in identities:
            raise ProtocolError(
                f'Duplicate forecast collection: {benchmark}/{collection} aliases {identities[identity]}')
        identities[identity] = f'{benchmark}/{collection}'
    return tuple(identities)

"""Right-edge-safe discrete wavelet forecasts and block observations."""

from dataclasses import dataclass

import numpy as np

from .baselines import BaselineKind, forecast_baseline
from .grid import positive_count
from .wavelets import _finite_series


@dataclass(frozen=True, eq=False)
class DWTBlock:
    # The rows can be timestamped earlier than the time at which they are known.
    available_at: int
    start: int
    bands: np.ndarray


def _wavelet():
    try:
        import pywt
    except ModuleNotFoundError as error:
        if error.name != "pywt":
            raise
        raise ImportError("Install industrial-forecasting[wavelets] to use discrete wavelets.") from error
    return pywt


def dwt_bands(history, *, wavelet="db3", levels=3):
    """Columns are the coarsest approximation and details from coarse to fine."""
    series = _finite_series(history)
    levels = positive_count(levels, "levels")
    pywt = _wavelet()
    basis = pywt.Wavelet(wavelet)
    if levels > pywt.dwt_max_level(len(series), basis.dec_len):
        raise ValueError("History is too short for the requested discrete wavelet levels.")
    coefficients = pywt.wavedec(series, basis, mode="symmetric", level=levels)
    return np.column_stack([
        pywt.waverec(
            [band if i == j else np.zeros_like(band) for j, band in enumerate(coefficients)],
            basis,
            mode="symmetric",
        )[:len(series)] for i in range(len(coefficients))
    ])


def right_edge_dwt(history, *, first_end, block_size, wavelet="db3", levels=3):
    """Recover each trailing block as it becomes available (zero-based indices)."""
    series = _finite_series(history)
    first_end = positive_count(first_end, "first_end")
    block_size = positive_count(block_size, "block_size")
    if not block_size <= first_end <= len(series):
        raise ValueError("The first exclusive end must contain a complete block within history.")
    blocks = []
    for end in range(first_end, len(series) + 1, block_size):
        bands = dwt_bands(series[:end], wavelet=wavelet, levels=levels)[-block_size:].copy()
        bands.setflags(write=False)
        blocks.append(DWTBlock(end - 1, end - block_size, bands))
    return tuple(blocks)


def predict_jin_kim(history, horizon, *, first_end, details=(), approximation_forecast=None, detail_forecast=None):
    """Forecast the available smooth band and selected db3 detail levels at aligned cuts."""
    series = _finite_series(history)
    horizon = positive_count(horizon, "horizon")
    first_end = positive_count(first_end, "first_end")
    details = tuple(details)
    if (horizon not in (2, 4) or (len(series) - first_end) % horizon
            or len(details) != len(set(details)) or any(type(i) is not int or not 1 <= i <= 3 for i in details)):
        raise ValueError("Use an aligned two- or four-step cutoff and unique detail levels 1–3.")
    blocks = right_edge_dwt(series, first_end=first_end, block_size=horizon)
    bands = np.concatenate([block.bands for block in blocks], axis=0)
    predictors = [(0, approximation_forecast or _trend)]
    predictors.extend((4 - level, detail_forecast or _last) for level in details)
    outputs = [_finite_series(fit_predict(bands[:, idx].copy(), horizon)) for idx, fit_predict in predictors]
    if any(output.shape != (horizon,) for output in outputs):
        raise ValueError("Each branch must forecast exactly the requested horizon.")
    return np.sum(outputs, axis=0)


def _last(history, horizon):
    return forecast_baseline(BaselineKind.LAST_VALUE, history, horizon)[0]


@dataclass(frozen=True)
class ArinoSpec:
    wavelet: str = "db3"
    levels: int = 3
    slow_bands: int = 1
    seasonal_period: int = 12
    padding: int = 0
    trend_lookback: int | None = None

    def __post_init__(self):
        if not isinstance(self.wavelet, str) or not self.wavelet:
            raise ValueError("Specify a discrete wavelet name.")
        levels = positive_count(self.levels, "levels")
        slow_bands = positive_count(self.slow_bands, "slow_bands")
        positive_count(self.seasonal_period, "seasonal_period")
        if self.trend_lookback is not None:
            positive_count(self.trend_lookback, "trend_lookback")
        if slow_bands > levels or type(self.padding) is not int or self.padding < 0:
            raise ValueError("Both branch groups must be nonempty and padding must be nonnegative.")


def arino_components(history, spec=ArinoSpec(), *, padding_forecast=None):
    """Compute one DWT, split bands, and drop every padded point before fitting."""
    series = _finite_series(history)
    if spec.padding:
        padding_forecast = padding_forecast or _trend
        tail = _finite_series(padding_forecast(series.copy(), spec.padding))
        if len(tail) != spec.padding:
            raise ValueError("Padding forecast does not match the requested length.")
        padded = np.concatenate((series, tail))
    else:
        padded = series
    center = padded.mean()
    bands = dwt_bands(padded - center, wavelet=spec.wavelet, levels=spec.levels)[:len(series)]
    return bands[:, :spec.slow_bands].sum(axis=1) + center, bands[:, spec.slow_bands:].sum(axis=1)


def _trend(history, horizon):
    return forecast_baseline(BaselineKind.LINEAR_TREND, history, horizon)[0]


def predict_arino(history, horizon, spec=ArinoSpec(), *, trend_forecast=None, seasonal_forecast=None,
                  padding_forecast=None):
    """Ariño-style independent branches; the default estimators are lightweight controls."""
    horizon = positive_count(horizon, "horizon")
    slow, seasonal = arino_components(history, spec, padding_forecast=padding_forecast)
    first = _finite_series((trend_forecast or _trend)(slow[-spec.trend_lookback:] if spec.trend_lookback
                                                      else slow, horizon))
    if seasonal_forecast is None:
        second = forecast_baseline(
            BaselineKind.SEASONAL_NAIVE, seasonal, horizon, seasonal_period=spec.seasonal_period,
        )[0]
    else:
        second = _finite_series(seasonal_forecast(seasonal, horizon))
    if first.shape != (horizon,) or second.shape != (horizon,):
        raise ValueError("Each branch must forecast exactly the requested horizon.")
    return first + second

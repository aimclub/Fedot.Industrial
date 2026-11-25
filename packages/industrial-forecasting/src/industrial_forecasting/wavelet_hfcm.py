"""Local Wavelet-HFCM variant with causal Haar bands and a recursive forecast."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .contracts import ForecastScenario, ModelScope, PointForecast
from .grid import positive_count
from .wavelets import _finite_series, causal_haar


@dataclass(frozen=True)
class WaveletHFCMSpec:
    levels: int = 2
    order: int = 3
    alpha: float = 0.1

    def __post_init__(self):
        positive_count(self.levels, "levels")
        object.__setattr__(self, "order", positive_count(self.order, "order"))
        if (isinstance(self.alpha, (bool, np.bool_)) or not isinstance(self.alpha, Real)
                or not np.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("alpha must be a finite positive number.")


@dataclass(frozen=True, eq=False)
class WaveletHFCM:
    """Fitted band dynamics; lag blocks are ordered oldest to newest."""

    _weights: np.ndarray
    _intercept: np.ndarray
    _tail: np.ndarray
    _center: float
    _scale: float

    def predict(self, horizon: int) -> np.ndarray:
        horizon = positive_count(horizon, "horizon")
        order, bands = self._tail.shape
        path = np.empty((order + horizon, bands))
        path[:order] = self._tail
        with np.errstate(over="raise", invalid="raise"):
            for step in range(horizon):
                past = path[step:step + order].reshape(-1)
                path[step + order] = np.tanh(past @ self._weights + self._intercept)
            forecast = self._center + self._scale * path[order:].sum(axis=1)
        return forecast


def _ridge(design, target, alpha):
    # Centering leaves the intercept unpenalized. Thin SVD also handles collinear bands.
    x_mean, y_mean = design.mean(axis=0), target.mean(axis=0)
    u, singular, vt = np.linalg.svd(design - x_mean, full_matrices=False)
    weights = (vt.T * (singular / (singular**2 + alpha))) @ (u.T @ (target - y_mean))
    return weights, y_mean - x_mean @ weights


def fit_wavelet_hfcm(history, spec: WaveletHFCMSpec = WaveletHFCMSpec()) -> WaveletHFCM:
    """Learn normalization and cross-band dynamics only from the supplied history."""
    series = _finite_series(history)
    lower, upper = float(series.min()), float(series.max())
    center = .5 * lower + .5 * upper
    scale = max(upper - center, center - lower) or 1.
    bands = causal_haar((series - center) / scale, spec.levels).values
    if len(bands) <= spec.order:
        raise ValueError("History must have enough supported Haar rows for at least one lagged training pair.")
    windows = np.lib.stride_tricks.sliding_window_view(bands, spec.order + 1, axis=0)
    design = windows[:, :, :-1].transpose(0, 2, 1).reshape(len(windows), -1)
    limit = np.nextafter(1., 0.)
    targets = np.arctanh(np.clip(windows[:, :, -1], -limit, limit))
    weights, intercept = _ridge(design, targets, spec.alpha)
    tail = bands[-spec.order:].copy()
    for value in (weights, intercept, tail):
        value.setflags(write=False)
    return WaveletHFCM(weights, intercept, tail, center, scale)


def predict_wavelet_hfcm(scenario: ForecastScenario, spec: WaveletHFCMSpec = WaveletHFCMSpec()) -> PointForecast:
    """Fit each regularly sampled series independently and preserve its forecast coordinates."""
    if scenario.scope is not ModelScope.LOCAL or scenario.covariates:
        raise ValueError("Wavelet-HFCM requires local scenarios without covariates.")
    steps = np.diff(np.concatenate((scenario.grid.history, scenario.grid.future)))
    same_step = (np.allclose(steps, steps[0], rtol=1e-9, atol=0.) if steps.dtype.kind == "f"
                 else np.all(steps == steps[0]))
    if not same_step or not np.isfinite(steps).all() or not steps[0] > np.zeros((), dtype=steps.dtype):
        raise ValueError("Wavelet-HFCM requires a constant time step across history and forecast.")
    values = np.stack([fit_wavelet_hfcm(row, spec).predict(scenario.horizon) for row in scenario.history])
    return PointForecast.for_scenario(scenario, values)

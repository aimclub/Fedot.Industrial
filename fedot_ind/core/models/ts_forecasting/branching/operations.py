"""Window-local representations and a parameter-free forecast mean.

    Representations are applied AFTER windowing. No target or future observation
    enters the window transformation, unlike smoothing an entire training series.
"""

import numpy as np


def _windows(features):
    values = np.asarray(features)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError('Forecast representations require a nonempty (samples, window) matrix')
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError('Forecast windows must be finite and real')
    return values


class CenterWindows:
    def __init__(self, params=None):
        pass

    def transform(self, features):
        values = _windows(features)
        return values - values.mean(axis=1, keepdims=True)


class DifferenceWindows:
    def __init__(self, params=None):
        pass

    def transform(self, features):
        values = _windows(features)
        return np.diff(values, axis=1, prepend=values[:, :1])


class SmoothWindows:
    def __init__(self, params=None):
        pass

    def transform(self, features):
        values = _windows(features)
        previous = np.concatenate((values[:, :1], values[:, :-1]), axis=1)
        return (values + previous) / 2.


class FeatureJoin:
    def __init__(self, params=None):
        pass

    def transform(self, features):
        return _windows(features).copy()


class MeanForecast:
    """Unweighted mean of equal horizon blocks; no fitted blending weights."""

    def __init__(self, params=None):
        self.horizon_ = None

    def fit(self, input_data):
        horizon = input_data.task.task_params.forecast_length
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError('A forecast mean requires a positive horizon')
        self.horizon_ = horizon
        self._blocks(input_data.features)
        return self

    def _blocks(self, features):
        values = _windows(features)
        if self.horizon_ is None:
            raise ValueError('Forecast mean has not been initialized')
        if values.shape[1] % self.horizon_ or values.shape[1] < 2 * self.horizon_:
            raise ValueError('Forecast mean needs at least two complete equal-horizon blocks')
        return values.reshape(len(values), -1, self.horizon_)

    def predict(self, input_data):
        if input_data.task.task_params.forecast_length != self.horizon_:
            raise ValueError('Forecast horizon changed after fitting')
        return self._blocks(input_data.features).mean(axis=1)

"""Localised Fourier pairs and a positional-feature forecasting control."""

import numpy as np

from .grid import positive_count
from .wavelets import _finite_series


def wire_features(coordinates, frequencies, widths, *, centers=0.):
    t = np.asarray(coordinates, dtype=float)
    omega, width, center = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, dtype=float))
                                                 for x in (frequencies, widths, centers)])
    if (t.ndim != 1 or omega.ndim != 1 or not len(omega)
            or not all(np.isfinite(x).all() for x in (t, omega, width, center)) or (width < 0).any()):
        raise ValueError("Use finite coordinate vectors and nonnegative envelope widths.")
    phase = (t[:, None] - center) * omega
    envelope = np.exp(-np.square((t[:, None] - center) * width))
    return np.concatenate((np.sin(phase) * envelope, np.cos(phase) * envelope), axis=1)


def predict_positional_wire(history, horizon, *, frequencies, widths, alpha=1., lags=4,
                            mode="wire", centers=0.):
    """Compare the same recursive ridge with localised, periodic or no position inputs."""
    x = _finite_series(history)
    horizon, lags = positive_count(horizon, "horizon"), positive_count(lags, "lags")
    if len(x) <= lags or not np.isfinite(alpha) or alpha <= 0 or mode not in {"wire", "fourier", "none"}:
        raise ValueError("Invalid history, ridge regularisation or positional mode.")
    coordinates = np.arange(lags, len(x) + horizon) / (len(x) - 1)
    pos = (np.empty((len(coordinates), 0)) if mode == "none" else
           wire_features(coordinates, frequencies, widths if mode == "wire" else 0., centers=centers))
    lagged = np.lib.stride_tricks.sliding_window_view(x, lags + 1)
    design = np.column_stack((lagged[:, :-1], pos[:-horizon]))
    mean, target_mean = design.mean(axis=0), lagged[:, -1].mean()
    centered = design - mean
    weights = np.linalg.solve(centered.T @ centered + alpha * np.eye(len(mean)),
                              centered.T @ (lagged[:, -1] - target_mean))
    path = np.concatenate((x[-lags:], np.empty(horizon)))
    for step in range(horizon):
        features = np.concatenate((path[step:step + lags], pos[-horizon + step]))
        path[step + lags] = target_mean + (features - mean) @ weights
    return path[lags:]

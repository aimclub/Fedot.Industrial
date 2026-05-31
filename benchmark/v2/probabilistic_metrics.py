from __future__ import annotations

import numpy as np


class ProbabilisticMetricError(ValueError):
    """Raised when a probabilistic metric cannot be computed."""


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """Pinball loss for a single quantile."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(actual) != len(predicted):
        raise ProbabilisticMetricError("Length mismatch")
    if not 0 < quantile < 1:
        raise ProbabilisticMetricError(f"Quantile {quantile} not in (0,1)")

    residual = actual - predicted
    loss = np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)
    return float(np.mean(loss))


def weighted_quantile_loss(
    y_true: np.ndarray,
    quantiles: dict[float, np.ndarray],
    weights: dict[float, float] | None = None,
) -> float:
    """Weighted Quantile Loss (WQL)."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)

    if weights is None:
        weights = {q: 1.0 for q in quantiles}

    total_loss = 0.0
    total_weight = 0.0
    for q, q_vals in quantiles.items():
        q_arr = np.asarray(q_vals, dtype=float).reshape(-1)
        if len(q_arr) != len(actual):
            raise ProbabilisticMetricError(f"Quantile {q} length mismatch")
        residual = actual - q_arr
        pinball = np.where(residual >= 0, q * residual, (q - 1) * residual)
        w = weights.get(q, 1.0)
        total_loss += w * np.mean(pinball)
        total_weight += w

    if total_weight < 1e-8:
        return 0.0
    return float(total_loss / total_weight)


def crps(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> float:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    sam = np.asarray(samples, dtype=float)

    if sam.ndim != 2:
        raise ProbabilisticMetricError(f"Samples must be 2D, got {sam.ndim}D")
    if sam.shape[1] != len(actual):
        raise ProbabilisticMetricError("Horizon mismatch")

    n_samples = sam.shape[0]
    abs_diff = np.abs(sam - actual)
    first_term = np.mean(abs_diff, axis=0)

    pairwise = 0.0
    for i in range(n_samples):
        for j in range(n_samples):
            pairwise += np.abs(sam[i] - sam[j])
    pairwise = pairwise / (n_samples * n_samples)
    second_term = 0.5 * pairwise

    crps_val = np.mean(first_term - second_term)
    return float(crps_val)


def interval_coverage(
    y_true: np.ndarray,
    intervals: dict[float, tuple[np.ndarray, np.ndarray]],
    nominal: float,
) -> float:
    """Empirical coverage for a given nominal interval."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    if nominal not in intervals:
        raise ProbabilisticMetricError(f"Nominal {nominal} not in intervals")
    lower, upper = intervals[nominal]
    lower_arr = np.asarray(lower, dtype=float).reshape(-1)
    upper_arr = np.asarray(upper, dtype=float).reshape(-1)
    if len(lower_arr) != len(actual) or len(upper_arr) != len(actual):
        raise ProbabilisticMetricError("Interval length mismatch")
    covered = (actual >= lower_arr) & (actual <= upper_arr)
    return float(np.mean(covered))


def calibration(
    y_true: np.ndarray,
    quantiles: dict[float, np.ndarray],
    quantile: float,
) -> float:
    """Empirical frequency (y_true <= predicted) for a given quantile."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    if quantile not in quantiles:
        raise ProbabilisticMetricError(f"Quantile {quantile} not in quantiles")
    pred = np.asarray(quantiles[quantile], dtype=float).reshape(-1)
    if len(pred) != len(actual):
        raise ProbabilisticMetricError("Length mismatch")
    return float(np.mean(actual <= pred))

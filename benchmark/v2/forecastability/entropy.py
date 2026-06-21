import numpy as np


import math
import numpy as np
from scipy.spatial import KDTree


def _is_valid(values: np.ndarray, min_length: int) -> bool:
    values = np.asarray(values, dtype=float)
    return len(values) >= min_length and np.all(np.isfinite(values))


def _embed(values: np.ndarray, dimension: int, delay: int = 1) -> np.ndarray:
    n = len(values) - (dimension - 1) * delay
    if n <= 0:
        return np.empty((0, dimension))
    return np.asarray([values[i:i + dimension * delay:delay] for i in range(n)])


def compute_sample_entropy(values: np.ndarray, m: int = 2, r: float | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if not _is_valid(values, m + 3):
        return float("nan")
    std = np.std(values)
    if std <= 0:
        return 0.0
    if r is None:
        r = 0.2 * std
    xm = _embed(values, m)
    xm1 = _embed(values, m + 1)
    if len(xm) < 2 or len(xm1) < 2:
        return float("nan")
    tree_m = KDTree(xm)
    tree_m1 = KDTree(xm1)
    count_m = np.sum([len(tree_m.query_ball_point(x, r, p=np.inf)) - 1 for x in xm])
    count_m1 = np.sum([len(tree_m1.query_ball_point(x, r, p=np.inf)) - 1 for x in xm1])
    if count_m <= 0:
        return float("inf")
    if count_m1 <= 0:
        return float("inf")
    return float(-np.log(count_m1 / count_m))


def compute_approximate_entropy(values: np.ndarray, m: int = 2, r: float | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if not _is_valid(values, m + 3):
        return float("nan")
    std = np.std(values)
    if std <= 0:
        return 0.0
    if r is None:
        r = 0.2 * std

    def _phi(dim: int) -> float:
        patterns = _embed(values, dim)
        if len(patterns) == 0:
            return float("nan")
        tree = KDTree(patterns)
        c = np.asarray([
            len(tree.query_ball_point(pattern, r, p=np.inf)) / len(patterns)
            for pattern in patterns
        ])
        c = np.maximum(c, 1e-12)
        return float(np.mean(np.log(c)))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return float("nan")

    return float(phi_m - phi_m1)


def compute_permutation_entropy(values: np.ndarray, dim: int = 3, delay: int = 1) -> float:
    values = np.asarray(values, dtype=float)
    if not _is_valid(values, dim + 2):
        return float("nan")
    embedded = _embed(values, dim, delay)
    if len(embedded) == 0:
        return float("nan")
    patterns = {}
    for row in embedded:
        key = tuple(np.argsort(row))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.asarray(list(patterns.values()), dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(math.factorial(dim))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def compute_spectral_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not _is_valid(values, 8):
        return float("nan")
    values = values - np.mean(values)
    spectrum = np.fft.rfft(values)
    power = np.abs(spectrum) ** 2
    if len(power) <= 1:
        return float("nan")
    power = power[1:]
    total_power = np.sum(power)
    if total_power <= 0:
        return 0.0
    probs = power / total_power
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(len(probs))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def compute_svd_entropy(values: np.ndarray, dim: int = 10) -> float:
    values = np.asarray(values, dtype=float)
    if not _is_valid(values, dim + 2):
        return float("nan")
    trajectory = _embed(values, dim)
    if len(trajectory) < 2:
        return float("nan")
    _, singular_values, _ = np.linalg.svd(trajectory, full_matrices=False)
    singular_values = singular_values[singular_values > 0]
    if len(singular_values) == 0:
        return 0.0
    probs = singular_values / np.sum(singular_values)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(len(probs))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)
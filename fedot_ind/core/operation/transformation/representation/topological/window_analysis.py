"""Explicitly bounded topology diagnostics for time series windows."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


def _cloud(points, limit):
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("Expected a finite (points, coordinates) cloud.")
    if len(x) > limit:
        raise ValueError("Point cloud exceeds the configured limit.")
    return x


def cknn_graph(points, *, k, delta=1., max_points=1024):
    """Boolean adjacency; a strict threshold leaves zero-radius points disconnected."""
    x = _cloud(points, max_points)
    if type(k) is not int or not 1 <= k < len(x) or not np.isfinite(delta) or delta <= 0:
        raise ValueError("CkNN requires 1 <= k < n and a positive finite delta.")
    dist = cdist(x, x)
    nonself = dist.copy()
    np.fill_diagonal(nonself, np.inf)
    radii = np.partition(nonself, k - 1, axis=1)[:, k - 1]
    thresholds = delta * np.sqrt(radii[:, None] * radii)
    adj = dist < thresholds
    np.fill_diagonal(adj, False)
    return adj


def delay_cloud(window, *, dimension, delay):
    x = np.asarray(window, dtype=float)
    if x.ndim not in (1, 2) or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite (time,) or (time, channels) observations.")
    if any(type(value) is not int or value < 1 for value in (dimension, delay)):
        raise ValueError("Embedding dimension and delay must be positive integers.")
    n = len(x) - (dimension - 1) * delay
    if n < 2:
        raise ValueError("Delay embedding needs at least two points.")
    return np.column_stack([x[i * delay:i * delay + n] for i in range(dimension)])


def _persistence(matrix, dim):
    from ripser import ripser
    return ripser(matrix, maxdim=dim, distance_matrix=True)["dgms"]


def persistence(points, *, max_dim=1, max_points=256):
    x = _cloud(points, max_points)
    if max_dim not in (0, 1):
        raise ValueError("Only H0 and H1 are supported.")
    return _persistence(cdist(x, x), max_dim)


def finite_lifetimes(diagram):
    diagram = np.asarray(diagram).reshape(-1, 2)
    valid = diagram[np.isfinite(diagram).all(axis=1)]
    return np.maximum(0., valid[:, 1] - valid[:, 0])


def _summary(diagrams):
    result = []
    for diagram in diagrams:
        lifetimes = finite_lifetimes(diagram)
        result.extend((len(lifetimes), lifetimes.sum(), lifetimes.max(initial=0.)))
    return np.asarray(result, dtype=float)


def subwindow_features(sequence, *, window, subwindow, stride, dimension=2, delay=1, max_points=256):
    """Mean and standard deviation of H0/H1 features over cached overlapping subwindows."""
    values = np.asarray(sequence, dtype=float)
    if (values.ndim not in (1, 2) or any(type(v) is not int or v < 1 for v in (window, subwindow, stride))
            or not subwindow <= window <= len(values)):
        raise ValueError("Expected valid nested window sizes and positive stride.")
    cache = {}
    result = []
    for start in range(0, len(values) - window + 1, stride):
        summaries = []
        for offset in range(start, start + window - subwindow + 1, stride):
            if offset not in cache:
                cloud = delay_cloud(values[offset:offset + subwindow], dimension=dimension, delay=delay)
                cache[offset] = _summary(persistence(cloud, max_points=max_points))
            summaries.append(cache[offset])
        result.append(np.concatenate((np.mean(summaries, axis=0), np.std(summaries, axis=0))))
    return np.asarray(result)


@dataclass(frozen=True, eq=False)
class LandscapeSeries:
    end_indices: np.ndarray
    norms: np.ndarray
    volatility: np.ndarray


def rolling_landscape(returns, *, window, stride=1, max_points=256):
    """L2 norm of all finite H1 landscapes, aligned to the last known observation."""
    x = np.asarray(returns, dtype=float)
    if (x.ndim != 2 or not np.isfinite(x).all() or type(window) is not int
            or type(stride) is not int or not 2 <= window <= len(x) or stride <= 0):
        raise ValueError("Expected finite (time, channels) observations and a valid window.")
    ends = np.arange(window - 1, len(x), stride)
    volatility, norms = [], []
    for end in ends:
        sample = x[end - window + 1:end + 1]
        lifetime = finite_lifetimes(persistence(sample, max_points=max_points)[1])
        # At each x the landscape levels permute the tents; integral of a squared tent is l^3/12.
        norms.append(np.sqrt(np.sum(lifetime**3) / 12))
        volatility.append(np.sqrt(np.mean(sample.var(axis=0))))
    return LandscapeSeries(ends, np.asarray(norms), np.asarray(volatility))


def cross_barcode(left, right, *, max_points=256):
    p, q = _cloud(left, max_points), _cloud(right, max_points)
    if p.shape[1] != q.shape[1] or len(p) + len(q) > max_points:
        raise ValueError("Clouds must share feature axes and fit the joint point budget.")
    joined = np.concatenate((p, q))
    distances = cdist(joined, joined)
    distances[len(p):, len(p):] = 0
    return _persistence(distances, 1)[1]


def manifold_topology_divergence(left, right, *, sample_size, repeats=4, seed=0, max_points=256):
    """Return the two directional mean lengths, without pooling their meanings."""
    p, q = _cloud(left, max_points), _cloud(right, max_points)
    if (type(sample_size) is not int or not 1 <= sample_size <= min(len(p), len(q))
            or type(repeats) is not int or repeats < 1):
        raise ValueError("Invalid cloud sampling parameters.")
    rng = np.random.default_rng(seed)
    totals = np.zeros(2)
    for _ in range(repeats):
        a, b = (x[rng.choice(len(x), size=sample_size, replace=False)] for x in (p, q))
        totals += [finite_lifetimes(cross_barcode(x, y, max_points=max_points)).sum()
                   for x, y in ((a, b), (b, a))]
    return tuple(totals / repeats)

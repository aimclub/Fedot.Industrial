"""Minimum stable prediction lengths from reverse nearest neighbours."""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, squareform

from .protocol import run_early, training_inputs


def _stable_lengths(neighbors, labels, groups, minimum_support=0.):
    cutoffs, count = neighbors.shape
    individual = np.full(count, cutoffs - 1, dtype=int)
    for i in range(count):
        if 1 / np.count_nonzero(labels == labels[i]) < minimum_support:
            continue
        reverse = neighbors == i
        unstable = np.flatnonzero((reverse != reverse[-1]).any(axis=1))
        individual[i] = unstable[-1] + 1 if len(unstable) else 0
    for group in groups:
        if len(group) <= 1 or len(set(labels[list(group)])) != 1:
            raise ValueError("Each group must combine at least two records of the same class.")
        if len(group) / np.count_nonzero(labels == labels[group[0]]) < minimum_support:
            continue
        in_group = np.isin(np.arange(count), group)
        stable_reverse = in_group[neighbors][:, ~in_group]
        stable_members = in_group[neighbors[:, in_group]].all(axis=1)
        invalid = np.flatnonzero(~stable_members | (stable_reverse != stable_reverse[-1]).any(axis=1))
        start = invalid[-1] + 1 if len(invalid) else 0
        if start < cutoffs:
            individual[in_group] = np.minimum(individual[in_group], start)
    return individual


def _homogeneous_clusters(distances, labels):
    """Single-link hierarchy in the full observation space."""
    n = len(labels)
    # linkage only consumes the off-diagonal distances.
    hierarchy = linkage(squareform(distances, checks=False), method="single")
    clusters, homogeneous = [(i,) for i in range(n)], []
    for left, right, _, _ in hierarchy:
        members = clusters[int(left)] + clusters[int(right)]
        clusters.append(members)
        if len(set(labels[list(members)])) == 1:
            homogeneous.append(members)
    return homogeneous


@dataclass(frozen=True, eq=False)
class EarlyNearestNeighbor:
    schedule: object
    train: np.ndarray
    labels: np.ndarray
    minimum_steps: np.ndarray

    def predict_one(self, series):
        def nearest(prefix):
            distances = ((self.train[:, :, :prefix.shape[-1]] - prefix)**2).sum(axis=(1, 2))
            return int(distances.argmin())

        def decide(prefix, step):
            i = nearest(prefix)
            return self.labels[i] if step >= self.minimum_steps[i] else None

        return run_early(series, self.schedule, decide, lambda prefix, _step: self.labels[nearest(prefix)])


def fit_early_nearest_neighbor(series, labels, schedule, *, merge_same_label=True,
                               minimum_support=0., max_records=512):
    """Compute stable reverse neighborhoods for leaves and homogeneous single-link clusters."""
    x, y = training_inputs(series, labels, schedule)
    if len(x) > max_records or not 0 <= minimum_support <= 1:
        raise ValueError("Invalid training panel size or minimum class support.")
    nearest = []
    for cutoff in schedule.cutoffs:
        rows = x[:, :, :cutoff].reshape(len(x), -1)
        dist = cdist(rows, rows, "sqeuclidean")
        np.fill_diagonal(dist, np.inf)
        nearest.append(dist.argmin(axis=1))
    neighbors = np.asarray(nearest)
    groups = _homogeneous_clusters(dist, y) if merge_same_label else ()
    steps = _stable_lengths(neighbors, y, groups, minimum_support)
    return EarlyNearestNeighbor(schedule, x.copy(), y.copy(), steps)

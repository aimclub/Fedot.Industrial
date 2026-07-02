"""Anchor selection strategies for Pairwise Difference Learning."""

from __future__ import annotations

import numpy as np

from .config import PairwiseLearningConfig
from .pair_targets import normalize_target_vector


def _evenly_spaced_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
    if len(indices) <= max_count:
        return indices.astype(int)
    positions = np.linspace(0, len(indices) - 1, num=max_count, dtype=int)
    return indices[positions].astype(int)


class ClassificationAdaptiveAnchorSelector:
    """Select training anchors for classification."""

    def __init__(self, config: PairwiseLearningConfig):
        self._config = config.normalized()

    def select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return anchor row indices into the training set.

        Uses ``pairing_policy`` and ``max_pairs`` to choose between the full training set and a per-class subsample.
        """
        # TODO: This is being removed now because X has never been used anywhere before, but it might come in handy in the future.
        del X
        target = normalize_target_vector(y).astype(int)
        n_samples = len(target)
        if n_samples == 0:
            raise ValueError("Cannot select PDL anchors for an empty target.")
        if (
            self._config.pairing_policy in {"all_pairs", "exact"}
            and n_samples * n_samples <= self._config.max_pairs
        ):
            return np.arange(n_samples, dtype=int)
        if n_samples * n_samples <= self._config.max_pairs:
            return np.arange(n_samples, dtype=int)

        selected: list[int] = []
        for label in np.unique(target):
            class_indices = np.flatnonzero(target == label)
            selected.extend(
                _evenly_spaced_indices(class_indices, self._config.anchors_per_class)
            )
        return np.asarray(sorted(set(selected)), dtype=int)


class RegressionEvenAnchorSelector:
    """Select training anchors for regression."""

    def __init__(self, config: PairwiseLearningConfig):
        self._config = config.normalized()

    def select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return anchor row indices spread across the training target range."""
        # TODO: This is being removed now because X has never been used anywhere before, but it might come in handy in the future.
        del X
        target_vector = normalize_target_vector(y)
        n_samples = len(target_vector)
        if n_samples == 0:
            raise ValueError("Cannot select PDL anchors for an empty target.")
        if (
            self._config.pairing_policy in {"all_pairs", "exact"}
            and n_samples * n_samples <= self._config.max_pairs
        ):
            return np.arange(n_samples, dtype=int)
        if n_samples * n_samples <= self._config.max_pairs:
            return np.arange(n_samples, dtype=int)

        max_anchor_count = max(
            1, min(n_samples, self._config.max_pairs // max(1, n_samples))
        )
        return _evenly_spaced_indices(np.arange(n_samples, dtype=int), max_anchor_count)

# TODO: in the future should be solver from string in config to class instance
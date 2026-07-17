"""Pair sampling strategies for Pairwise Difference Learning."""

from __future__ import annotations

import numpy as np


class AllPairsSampler:
    """Materialize every (left, anchor) combination via a cross-product."""

    def sample(
        self, n_left: int, anchor_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return pair_indices(n_left, anchor_indices)


def pair_indices(
    n_left: int, anchor_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned left/anchor index arrays for the full cross-product."""
    left_indices = np.repeat(np.arange(n_left, dtype=int), len(anchor_indices))
    repeated_anchor_indices = np.tile(anchor_indices.astype(int), n_left)
    return left_indices, repeated_anchor_indices

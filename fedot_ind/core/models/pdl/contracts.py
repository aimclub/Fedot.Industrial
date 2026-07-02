"""Protocol contracts for PDL strategy components."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PairFeatureBuilder(Protocol):
    def build(self, left: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        ...


class PairTargetBuilder(Protocol):
    def build(self, y_left: np.ndarray, y_anchor: np.ndarray) -> np.ndarray:
        ...


class AnchorSelector(Protocol):
    def select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...


class PairSampler(Protocol):
    def sample(
        self, n_left: int, anchor_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class PairAggregator(Protocol):
    def aggregate(
        self,
        pair_predictions: np.ndarray,
        anchor_labels: np.ndarray,
        *,
        n_classes: int,
    ) -> np.ndarray:
        ...


class UncertaintyEstimator(Protocol):
    def estimate(
        self,
        pair_predictions: np.ndarray,
        aggregated: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ...

"""Pair target builders for Pairwise Difference Learning."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_target_vector(target: Any) -> np.ndarray:
    return np.asarray(target).reshape(-1)


class ClassificationDissimilarityTargetBuilder:
    """Build classification pair targets: 0=same class, 1=different."""

    def build(self, y_left: np.ndarray, y_anchor: np.ndarray) -> np.ndarray:
        return (y_left != y_anchor).astype(int)


class RegressionDeltaLeftMinusAnchorTargetBuilder:
    """Build regression pair targets as ``target_left - target_anchor``."""

    def build(self, y_left: np.ndarray, y_anchor: np.ndarray) -> np.ndarray:
        return (y_left - y_anchor).astype(float)

# TODO: in the future should be solver from string in config to class instance

"""Diagnostics helpers for Pairwise Difference Learning."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import (
    CLASSIFICATION_DIFFERENT_LABEL,
    CLASSIFICATION_SAME_LABEL,
    REGRESSION_DELTA_SIGN,
    PairwiseLearningConfig,
)
from .pair_features import resolve_pairwise_backend


def pair_target_semantics(*, task: str) -> dict[str, Any]:
    """Describe the active pair-target contract for diagnostics and tooling."""
    if task == "classification":
        return {
            "task": "classification",
            "same_label": CLASSIFICATION_SAME_LABEL,
            "different_label": CLASSIFICATION_DIFFERENT_LABEL,
            "target_type": "dissimilarity",
            "target_formula": "int(left_class != anchor_class)",
            "inference_output": "same_probability", # P(same_label)
        }
    if task == "regression":
        return {
            "task": "regression",
            "delta_sign": REGRESSION_DELTA_SIGN,
            "target_formula": "target_left - target_anchor",
            "inference_reconstruction": "anchor_target + predicted_delta",
        }
    raise ValueError(f"Unsupported PDL task={task!r}.")


def pair_diagnostics(
    *,
    config: PairwiseLearningConfig,
    n_left: int,
    n_anchors: int,
    pair_feature_dim: int,
    anchor_indices: np.ndarray,
    task: str,
) -> dict[str, Any]:   # TODO: add typed contract diagnostic
    """Build the runtime diagnostics payload attached to each ``PairwiseBatch``."""
    backend_name, _ = resolve_pairwise_backend(config.backend)
    return {
        "backend": backend_name,
        "pairing_policy": config.pairing_policy,
        "n_train": int(n_left),
        "n_anchors": int(n_anchors),
        "n_pairs": int(n_left * n_anchors),
        "pair_feature_dim": int(pair_feature_dim),
        "anchor_indices": [int(index) for index in anchor_indices],
        "config": config.to_dict(),
        "pair_target_semantics": pair_target_semantics(task=task),
    }

"""Core utilities for Pairwise Difference Learning (PDL).

This module is a backward-compatible facade over the strategy-based PDL
implementation. Estimator facades live in ``pairwise_model``.

Pair-target contracts (kept backward compatible):
    * Classification: ``0 = same`` (``CLASSIFICATION_SAME_LABEL``),
      ``1 = different`` (``CLASSIFICATION_DIFFERENT_LABEL``). The training
      target is a *dissimilarity* label and inference returns the probability
      of the ``same`` class.
    * Regression: ``delta = target_left - target_anchor``
      (``REGRESSION_DELTA_SIGN = "left_minus_anchor"``); predictions are
      reconstructed as ``anchor_target + predicted_delta``.
"""

from __future__ import annotations
from typing import Any, Iterable
import numpy as np

from .config import (
    AGGREGATION_POLICY_ALIASES,
    CLASSIFICATION_DIFFERENT_LABEL,
    CLASSIFICATION_SAME_LABEL,
    PRIOR_EPS,
    REGRESSION_DELTA_SIGN,
    SUPPORTED_AGGREGATION_POLICIES,
    SUPPORTED_PAIR_FEATURE_MODES,
    PairwiseBatch,
    PairwiseLearningConfig,
    resolve_aggregation_policy,
)
from .diagnostics import pair_diagnostics
from .pair_features import (
    build_pair_features,
    normalize_feature_matrix,
    resolve_pairwise_backend,
    # pair_feature_dim
)
from .pair_features import torch  # noqa: F401 - re-exported for backward compatibility
from .pair_targets import normalize_target_vector
from .strategies import PDLStrategies, resolve_pdl_strategies

__all__ = [
    "AGGREGATION_POLICY_ALIASES",
    "CLASSIFICATION_DIFFERENT_LABEL",
    "CLASSIFICATION_SAME_LABEL",
    "PRIOR_EPS",
    "REGRESSION_DELTA_SIGN",
    "SUPPORTED_AGGREGATION_POLICIES",
    "SUPPORTED_PAIR_FEATURE_MODES",
    "PDLStrategies",
    "PairwiseBatch",
    "PairwiseLearningConfig",
    "build_pair_batch",
    "build_pair_features",
    "normalize_feature_matrix",
    "normalize_target_vector",
    # "pair_feature_dim",
    "predict_regression_by_chunks",
    "predict_similarity_by_chunks",
    "resolve_aggregation_policy",
    "resolve_pairwise_backend",
    "resolve_pdl_strategies",
    "torch",
]


def build_pair_batch(
    features: Any,
    target: Any,
    anchor_indices: Iterable[int],
    config: PairwiseLearningConfig,
    *,
    task: str,
    strategies: PDLStrategies | None = None,
) -> PairwiseBatch:
    """Build a training batch by composing the resolved PDL strategies.

    This is the central orchestration point for pair construction: sampler,
    feature builder, target builder and diagnostics are all applied here.
    """
    resolved = strategies or resolve_pdl_strategies(config, task=task)
    feature_matrix = normalize_feature_matrix(features)
    if task == "classification":
        target_vector = normalize_target_vector(target).astype(int)
    else:
        target_vector = normalize_target_vector(target).astype(float)
    anchors = np.asarray(tuple(anchor_indices), dtype=int)
    left_indices, repeated_anchor_indices = resolved.pair_sampler.sample(
        len(feature_matrix), anchors
    )
    pair_features = resolved.pair_feature_builder.build(
        feature_matrix, feature_matrix[anchors]
    )
    pair_target = resolved.pair_target_builder.build(
        target_vector[left_indices],
        target_vector[repeated_anchor_indices],
    )
    diagnostics = pair_diagnostics(
        config=resolved.config,
        n_left=len(feature_matrix),
        n_anchors=len(anchors),
        pair_feature_dim=pair_features.shape[1],
        anchor_indices=anchors,
        task=task,
    )
    return PairwiseBatch(
        pair_features,
        pair_target,
        left_indices,
        repeated_anchor_indices,
        diagnostics,
    )


# TODO: left over from the previous version, used in paiwise_model in two places
def predict_similarity_by_chunks(
    base_model: Any,
    features: Any,
    anchor_features: Any,
    config: PairwiseLearningConfig,
) -> np.ndarray:
    """Predict P(same) for every query/anchor pair, processing rows in chunks."""
    feature_matrix = normalize_feature_matrix(features)
    anchors = normalize_feature_matrix(anchor_features)
    n_samples = len(feature_matrix)
    n_anchors = len(anchors)
    if n_anchors == 0:
        raise ValueError("PDL prediction requires at least one anchor.")
    rows_per_chunk = max(1, int(config.chunk_size) // max(1, n_anchors))
    chunks = []
    for start in range(0, n_samples, rows_per_chunk):
        chunk = feature_matrix[start : start + rows_per_chunk]
        pair_features = build_pair_features(chunk, anchors, config)
        same_probability = _predict_same_probability(base_model, pair_features)
        chunks.append(same_probability.reshape(len(chunk), n_anchors))
    return np.vstack(chunks) if chunks else np.empty((0, n_anchors), dtype=float)

# TODO: left over from the previous version, used in paiwise_model in one place (prediction in the regression model)
def predict_regression_by_chunks(
    base_model: Any,
    features: Any,
    anchor_features: Any,
    anchor_target: Any,
    config: PairwiseLearningConfig,
) -> np.ndarray:
    """Predict regression targets by averaging anchor+delta reconstructions."""
    feature_matrix = normalize_feature_matrix(features)
    anchors = normalize_feature_matrix(anchor_features)
    anchor_target = normalize_target_vector(anchor_target).astype(float)
    n_samples = len(feature_matrix)
    n_anchors = len(anchors)
    rows_per_chunk = max(1, int(config.chunk_size) // max(1, n_anchors))
    predictions = []
    for start in range(0, n_samples, rows_per_chunk):
        chunk = feature_matrix[start : start + rows_per_chunk]
        pair_features = build_pair_features(chunk, anchors, config)
        deltas = np.asarray(base_model.predict(pair_features), dtype=float).reshape(
            len(chunk), n_anchors
        )
        predictions.append(anchor_target.reshape(1, -1) + deltas)
    return (
        np.concatenate([chunk.mean(axis=1) for chunk in predictions])
        if predictions
        else np.empty(0, dtype=float)
    )

# Helper for predict_similarity_by_chunks()
def _predict_same_probability(base_model: Any, pair_features: np.ndarray) -> np.ndarray:
    """Extract P(same) from a pairwise classifier's outputs."""
    if hasattr(base_model, "predict_proba"):
        probability = np.asarray(base_model.predict_proba(pair_features), dtype=float)
        classes = np.asarray(
            getattr(base_model, "classes_", np.arange(probability.shape[1]))
        )
        same_columns = np.flatnonzero(classes == CLASSIFICATION_SAME_LABEL)
        if len(same_columns) == 0:
            return np.zeros(pair_features.shape[0], dtype=float)
        return probability[:, int(same_columns[0])]
    prediction = np.asarray(base_model.predict(pair_features)).reshape(-1)
    return (prediction == CLASSIFICATION_SAME_LABEL).astype(float)

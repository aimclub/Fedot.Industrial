"""Classification aggregation strategies for Pairwise Difference Learning."""

from __future__ import annotations

import numpy as np

from .config import PairwiseLearningConfig
from .pair_targets import normalize_target_vector


class MeanSimilarityAggregator:
    """Aggregate P(same) into class probabilities by mean similarity per class."""

    def aggregate(
        self,
        pair_predictions: np.ndarray,
        anchor_labels: np.ndarray,
        *,
        n_classes: int,
    ) -> np.ndarray:
        """Convert an (n_samples, n_anchors) similarity matrix into class proba."""
        similarity_matrix = np.asarray(pair_predictions, dtype=float)
        anchor_labels = normalize_target_vector(anchor_labels).astype(int)
        probabilities = np.zeros((similarity_matrix.shape[0], n_classes), dtype=float)
        for class_id in range(n_classes):
            mask = anchor_labels == class_id
            if np.any(mask):
                probabilities[:, class_id] = np.mean(similarity_matrix[:, mask], axis=1)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        empty_rows = row_sums.squeeze(axis=1) <= np.finfo(float).eps
        if np.any(empty_rows):
            probabilities[empty_rows, :] = 1.0 / max(1, n_classes)
            row_sums = probabilities.sum(axis=1, keepdims=True)
        return probabilities / row_sums


def _ensure_supported_aggregation_config(config: PairwiseLearningConfig) -> None:
    config = config.normalized()
    if config.aggregation_policy != "mean_similarity":
        raise NotImplementedError(
            "aggregation_policy='paper_posterior' and 'weighted_posterior' "
            "are reserved for PR3."
        )
    if config.symmetric_inference:
        raise NotImplementedError("symmetric_inference=True is reserved for PR3.")


def resolve_pair_aggregator(config: PairwiseLearningConfig) -> MeanSimilarityAggregator:
    """Return the classification aggregator for the normalized config, it also normalized inside (for proof)."""
    _ensure_supported_aggregation_config(config)
    return MeanSimilarityAggregator()


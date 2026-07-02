"""Configuration and data containers for Pairwise Difference Learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

SUPPORTED_PAIR_FEATURE_MODES = ("concat_diff", "concat_absdiff", "diff_only")
CLASSIFICATION_SAME_LABEL = 0
CLASSIFICATION_DIFFERENT_LABEL = 1
REGRESSION_DELTA_SIGN = "left_minus_anchor"
SUPPORTED_AGGREGATION_POLICIES = (
    "mean_similarity",
    "paper_posterior",
    "weighted_posterior",
)
AGGREGATION_POLICY_ALIASES = {
    "posterior": "paper_posterior",
    "mean": "mean_similarity",
    "weighted": "weighted_posterior",
}
# TODO: do not use now
PRIOR_EPS = 1e-12


def resolve_aggregation_policy(raw: str) -> str:
    """Normalize user-facing aggregation policy names and aliases."""
    key = str(raw).strip().lower()
    canonical = AGGREGATION_POLICY_ALIASES.get(key, key)
    if canonical not in SUPPORTED_AGGREGATION_POLICIES:
        raise ValueError(
            f"Unsupported aggregation_policy={raw!r}. "
            f"Supported policies: {SUPPORTED_AGGREGATION_POLICIES}. "
            f"Aliases: {sorted(AGGREGATION_POLICY_ALIASES)}."
        )
    return canonical


@dataclass(frozen=True)
class PairwiseLearningConfig:
    """Immutable configuration for PDL training, inference and strategy resolution."""

    backend: str = "auto"
    pairing_policy: str = "adaptive_anchors"
    max_pairs: int = 250_000
    anchors_per_class: int = 20
    pair_feature_mode: str = "concat_diff"
    chunk_size: int = 8192
    random_state: int = 42
    aggregation_policy: str = "mean_similarity"
    symmetric_inference: bool = False
    class_prior_mode: str = "empirical"

    def normalized(self) -> "PairwiseLearningConfig":
        """Validate fields, resolve aliases and return a canonical config copy."""
        if self.max_pairs <= 0:
            raise ValueError("max_pairs must be positive.")
        if self.anchors_per_class <= 0:
            raise ValueError("anchors_per_class must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.pair_feature_mode not in SUPPORTED_PAIR_FEATURE_MODES:
            raise ValueError(
                f"Unsupported pair_feature_mode={self.pair_feature_mode!r}. "
                f"Supported modes: {SUPPORTED_PAIR_FEATURE_MODES}."
            )
        if self.pairing_policy not in {
            "adaptive_anchors",
            "all_pairs",
            "exact",
            "stratified_anchors",
        }:
            raise ValueError(
                f"Unsupported pairing_policy={self.pairing_policy!r}.")
        resolved_policy = resolve_aggregation_policy(self.aggregation_policy)
        if self.class_prior_mode not in {"empirical", "uniform"}:
            raise ValueError(
                f"Unsupported class_prior_mode={self.class_prior_mode!r}. "
                f"Supported modes: 'empirical', 'uniform'."
            )
        # if not isinstance(self.symmetric_inference, bool):
        # This condition can check bool, int and float values (eg. True == 1 == 1.0)
        if not (self.symmetric_inference == True or self.symmetric_inference == False):
            raise ValueError(
                f"symmetric_inference must be bool, got {type(self.symmetric_inference).__name__}."
            )
        return replace(self, aggregation_policy=resolved_policy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseBatch:
    """Training batch produced by pair construction."""

    features: np.ndarray
    target: np.ndarray
    left_indices: np.ndarray
    anchor_indices: np.ndarray
    # TODO: replace ``diagnostics`` with a typed contract.
    diagnostics: dict[str, Any]

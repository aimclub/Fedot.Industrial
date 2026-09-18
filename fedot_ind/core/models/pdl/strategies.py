"""Strategy resolution for Pairwise Difference Learning."""

from __future__ import annotations

from dataclasses import dataclass

from .aggregators import resolve_pair_aggregator
from .anchors import ClassificationAdaptiveAnchorSelector, RegressionEvenAnchorSelector
from .config import PairwiseLearningConfig
from .contracts import (
    AnchorSelector,
    PairAggregator,
    PairFeatureBuilder,
    PairSampler,
    PairTargetBuilder,
)
from .pair_features import resolve_pair_feature_builder
from .pair_targets import (
    ClassificationDissimilarityTargetBuilder,
    RegressionDeltaLeftMinusAnchorTargetBuilder,
)
from .samplers import AllPairsSampler


def _ensure_supported_symmetric_inference(config: PairwiseLearningConfig) -> PairwiseLearningConfig:
    config = config.normalized()
    if config.symmetric_inference:
        raise NotImplementedError(
            "symmetric_inference=True is reserved for future PR (PR 3).")
    return config


@dataclass(frozen=True)
class PDLStrategies:
    """Resolved bundle of strategy objects for one PDL task."""

    config: PairwiseLearningConfig
    task: str
    anchor_selector: AnchorSelector
    pair_feature_builder: PairFeatureBuilder
    pair_target_builder: PairTargetBuilder
    pair_sampler: PairSampler
    pair_aggregator: PairAggregator | None

    @property
    def random_state(self) -> int:
        return self.config.random_state


def resolve_pdl_strategies(
    config: PairwiseLearningConfig,
    *,
    task: str,
) -> PDLStrategies:
    """Resolve default strategies for classification or regression."""
    normalized = _ensure_supported_symmetric_inference(config)
    pair_sampler = AllPairsSampler()
    pair_feature_builder = resolve_pair_feature_builder(normalized)

    if task == "classification":
        pair_aggregator = resolve_pair_aggregator(normalized)
        return PDLStrategies(
            config=normalized,
            task=task,
            anchor_selector=ClassificationAdaptiveAnchorSelector(normalized),
            pair_feature_builder=pair_feature_builder,
            pair_target_builder=ClassificationDissimilarityTargetBuilder(),
            pair_sampler=pair_sampler,
            pair_aggregator=pair_aggregator,
        )
    if task == "regression":
        return PDLStrategies(
            config=normalized,
            task=task,
            anchor_selector=RegressionEvenAnchorSelector(normalized),
            pair_feature_builder=pair_feature_builder,
            pair_target_builder=RegressionDeltaLeftMinusAnchorTargetBuilder(),
            pair_sampler=pair_sampler,
            pair_aggregator=None,
        )
    raise ValueError(f"Unsupported PDL task={task!r}.")

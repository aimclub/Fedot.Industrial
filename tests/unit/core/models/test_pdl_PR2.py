"""Unit tests for PDL strategy resolution and backward compatibility (PR2)."""

import numpy as np
import pytest

from fedot_ind.core.models.pdl.aggregators import MeanSimilarityAggregator
from fedot_ind.core.models.pdl.anchors import (
    ClassificationAdaptiveAnchorSelector,
    RegressionEvenAnchorSelector,
)
from fedot_ind.core.models.pdl.config import PairwiseLearningConfig
from fedot_ind.core.models.pdl.pair_features import (
    ConcatAbsdiffPairFeatureBuilder,
    ConcatDiffPairFeatureBuilder,
    DiffOnlyPairFeatureBuilder,
    resolve_pair_feature_builder,
)
from fedot_ind.core.models.pdl.pair_targets import (
    ClassificationDissimilarityTargetBuilder,
    RegressionDeltaLeftMinusAnchorTargetBuilder,
)
from fedot_ind.core.models.pdl.pairwise_core import (
    build_pair_batch,
    resolve_pdl_strategies,
)
from fedot_ind.core.models.pdl.samplers import AllPairsSampler


class TestAggregators:
    def test_mean_similarity_aggregator_matches_legacy_function(self):
        similarity = np.array([[0.9, 0.1, 0.2], [0.3, 0.8, 0.7]])
        anchor_labels = np.array([0, 1, 1])

        strategy = MeanSimilarityAggregator().aggregate(similarity, anchor_labels, n_classes=2)
        np.testing.assert_allclose(strategy, np.array([[0.9 , 0.15],[0.3 , 0.75]]) / 1.05)

    def test_non_default_aggregation_policy_raises_until_pr3(self):
        config = PairwiseLearningConfig(aggregation_policy="paper_posterior")
        with pytest.raises(NotImplementedError, match="reserved for PR3"):
            resolve_pdl_strategies(config, task="classification")


class TestErrorsInConfig:
    def test_error_pair_feature_mode(self):
        config = PairwiseLearningConfig(pair_feature_mode='aaaa')
        with pytest.raises(ValueError, match='Unsupported pair_feature_mode='):
            config.normalized()
    
    def test_error_pairing_policy(self):
        config = PairwiseLearningConfig(pairing_policy='aaaa')
        with pytest.raises(ValueError, match='Unsupported pairing_policy='):
            config.normalized()
    
    def test_error_aggregation_policy(self):
        config = PairwiseLearningConfig(aggregation_policy='aaaa')
        with pytest.raises(ValueError, match='Unsupported aggregation_policy='):
            config.normalized()

    def test_error_symmetric_inference_1(self):
        config = PairwiseLearningConfig(symmetric_inference=1.1)
        with pytest.raises(ValueError, match='symmetric_inference must be bool, got '):
            config.normalized()
    
    def test_error_symmetric_inference_2(self):
        config = PairwiseLearningConfig(symmetric_inference=1.0)
        res = config.normalized()
        assert res.symmetric_inference==1


class TestPairFeatures:
    def test_pair_feature_builder_resolution_matches_legacy_modes(self):
        config_concat = PairwiseLearningConfig(pair_feature_mode="concat_diff", backend="numpy")
        config_abs = PairwiseLearningConfig(pair_feature_mode="concat_absdiff", backend="numpy")
        config_diff = PairwiseLearningConfig(pair_feature_mode="diff_only", backend="numpy")

        assert isinstance(resolve_pair_feature_builder(config_concat), ConcatDiffPairFeatureBuilder)
        assert isinstance(resolve_pair_feature_builder(config_abs), ConcatAbsdiffPairFeatureBuilder)
        assert isinstance(resolve_pair_feature_builder(config_diff), DiffOnlyPairFeatureBuilder)

    def test_unknown_builder(self):
        config = PairwiseLearningConfig(pair_feature_mode="aaa", backend="numpy")
        with pytest.raises(ValueError, match='Unsupported pair_feature_mode='):
            resolve_pair_feature_builder(config)


class TestSamplers:
    def test_all_pairs_sampler_matches_legacy_pair_indices(self):
        anchors = np.array([1, 3, 5], dtype=int)
        sampler_left, sampler_anchor = AllPairsSampler().sample(4, anchors)

        np.testing.assert_array_equal(sampler_left, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(sampler_anchor, [1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5])


class TestStrategies:
    def test_resolve_default_strategies_classification(self):
        config = PairwiseLearningConfig()
        strategies = resolve_pdl_strategies(config, task="classification")

        assert strategies.task == "classification"
        assert isinstance(strategies.anchor_selector, ClassificationAdaptiveAnchorSelector)
        assert isinstance(strategies.pair_target_builder, ClassificationDissimilarityTargetBuilder)
        assert isinstance(strategies.pair_sampler, AllPairsSampler)
        assert isinstance(strategies.pair_aggregator, MeanSimilarityAggregator)
        assert strategies.random_state == config.random_state

    def test_resolve_default_strategies_regression(self):
        config = PairwiseLearningConfig()
        strategies = resolve_pdl_strategies(config, task="regression")

        assert strategies.task == "regression"
        assert isinstance(strategies.anchor_selector, RegressionEvenAnchorSelector)
        assert isinstance(strategies.pair_target_builder, RegressionDeltaLeftMinusAnchorTargetBuilder)
        assert isinstance(strategies.pair_sampler, AllPairsSampler)
        assert strategies.pair_aggregator is None

    def test_error_symmetric_inference_for_PR3(self):
        config = PairwiseLearningConfig(symmetric_inference=True)
        with pytest.raises(NotImplementedError, match='symmetric_inference=True is reserved'):
            resolve_pdl_strategies(config, task="classification")

    def test_error_unknown_task(self):
        config = PairwiseLearningConfig()
        with pytest.raises(ValueError, match='Unsupported PDL task='):
            resolve_pdl_strategies(config, task="class")

    def test_random_state_is_available_on_strategy_context_for_future_stochastic_strategies(self):
        config = PairwiseLearningConfig(random_state=123)
        strategies = resolve_pdl_strategies(config, task="classification")
        assert strategies.random_state == 123
        assert strategies.config.random_state == 123


class TestBuildPairBatch:
    def test_build_classification_pairs_facade_matches_strategy_batch(self):
        features = np.array([[0.0], [1.0], [2.0]])
        target = np.array([5, 5, 7])
        anchors = np.array([0, 1, 2])
        config = PairwiseLearningConfig(backend="numpy", max_pairs=20)
        batch = build_pair_batch(
            features,
            target,
            anchors,
            config,
            task="classification",
            strategies=resolve_pdl_strategies(config, task="classification"),
        )
        np.testing.assert_array_equal(batch.features, [[0., 0.,  0.],
                                                       [0., 1., -1.],
                                                       [0., 2., -2.],
                                                       [1., 0.,  1.],
                                                       [1., 1.,  0.],
                                                       [1., 2., -1.],
                                                       [2., 0.,  2.],
                                                       [2., 1.,  1.],
                                                       [2., 2.,  0.]])
        np.testing.assert_array_equal(batch.target, [0, 0, 1, 0, 0, 1, 1, 1, 0])

    def test_build_regression_pairs_facade_matches_strategy_batch(self):
        features = np.array([[0.0], [1.0]])
        target = np.array([1.0, 3.0])
        anchors = np.array([0, 1])
        config = PairwiseLearningConfig(backend="numpy", max_pairs=20)
        batch = build_pair_batch(
            features,
            target,
            anchors,
            config,
            task="regression",
            strategies=resolve_pdl_strategies(config, task="regression"),
        )
        np.testing.assert_array_equal(batch.features, [[0., 0.,  0.],
                                                       [0., 1., -1.],
                                                       [1., 0.,  1.],
                                                       [1., 1.,  0.]])
        np.testing.assert_array_equal(batch.target, [0., -2., 2., 0.])

    def test_pair_target_semantics_preserved_after_strategy_refactor(self):
        features = np.array([[0.0], [1.0]])
        config = PairwiseLearningConfig(backend="numpy", max_pairs=20)

        clf_batch = build_pair_batch(
            features,
            np.array([0, 1]),
            np.array([0, 1]),
            config,
            task="classification",
        )
        reg_batch = build_pair_batch(
            features,
            np.array([1.0, 3.0]),
            np.array([0, 1]),
            config,
            task="regression",
        )

        clf_semantics = clf_batch.diagnostics["pair_target_semantics"]
        assert clf_semantics["same_label"] == 0
        assert clf_semantics["different_label"] == 1

        reg_semantics = reg_batch.diagnostics["pair_target_semantics"]
        assert reg_semantics["delta_sign"] == "left_minus_anchor"



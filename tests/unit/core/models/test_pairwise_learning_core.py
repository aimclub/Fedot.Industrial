import numpy as np
import pytest

from fedot_ind.core.models.pdl.pairwise_core import (
    PairwiseLearningConfig,
    build_classification_pairs,
    build_pair_features,
    build_regression_pairs,
    select_classification_anchor_indices,
    torch,
)


def test_numpy_and_torch_pair_builders_match_on_small_arrays():
    if torch is None:
        pytest.skip("torch is not available")
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    anchors = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)

    numpy_pairs = build_pair_features(
        features,
        anchors,
        PairwiseLearningConfig(backend="numpy", pair_feature_mode="concat_diff"),
    )
    torch_pairs = build_pair_features(
        features,
        anchors,
        PairwiseLearningConfig(backend="cpu", pair_feature_mode="concat_diff"),
    )

    assert numpy_pairs.shape == (4, 6)
    np.testing.assert_allclose(torch_pairs, numpy_pairs)


def test_adaptive_anchor_selection_is_deterministic_and_keeps_each_class():
    target = np.array([10, 10, 10, 20, 20, 20, 30, 30, 30])
    config = PairwiseLearningConfig(max_pairs=8, anchors_per_class=1)

    first = select_classification_anchor_indices(target, config)
    second = select_classification_anchor_indices(target, config)

    np.testing.assert_array_equal(first, second)
    assert set(target[first]) == {10, 20, 30}


def test_classification_pair_targets_support_labels_not_starting_at_zero():
    features = np.array([[0.0], [1.0], [2.0]])
    target = np.array([10, 20, 10])
    config = PairwiseLearningConfig(backend="numpy", max_pairs=20)

    batch = build_classification_pairs(features, target, np.array([0, 1, 2]), config)

    np.testing.assert_array_equal(
        batch.target,
        np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]),
    )
    assert batch.features.shape == (9, 3)
    assert batch.diagnostics["n_pairs"] == 9


def test_regression_pair_target_uses_left_minus_anchor_sign_convention():
    features = np.array([[0.0], [1.0]])
    target = np.array([1.0, 3.0])
    config = PairwiseLearningConfig(backend="numpy", max_pairs=20)

    batch = build_regression_pairs(features, target, np.array([0, 1]), config)

    np.testing.assert_allclose(batch.target, np.array([0.0, -2.0, 2.0, 0.0]))
    np.testing.assert_allclose(
        batch.features,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

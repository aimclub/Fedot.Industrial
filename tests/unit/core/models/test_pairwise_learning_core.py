import numpy as np
import pytest

from fedot_ind.core.models.pdl.pairwise_core import (
    PairwiseLearningConfig,
    aggregate_similarity_to_class_proba,
    build_classification_pairs,
    build_pair_batch,
    build_pair_features,
    build_regression_pairs,
    pair_feature_dim,
    select_classification_anchor_indices,
    select_regression_anchor_indices,
    _predict_same_probability,
    torch,
)
from fedot_ind.core.models.pdl import pair_features as pair_features_module
from fedot_ind.core.models.pdl.anchors import ClassificationAdaptiveAnchorSelector
from fedot_ind.core.models.pdl.diagnostics import pair_target_semantics
from fedot_ind.core.models.pdl.pair_features import (
    _build_pair_features_torch,
    normalize_feature_matrix,
    resolve_pairwise_backend,
)


def test_numpy_and_torch_pair_builders_match_on_small_arrays():
    if torch is None:
        pytest.skip("torch is not available")
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    anchors = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)

    numpy_pairs = build_pair_features(
        features,
        anchors,
        PairwiseLearningConfig(
            backend="numpy", pair_feature_mode="concat_diff"),
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

    first = ClassificationAdaptiveAnchorSelector(
        config).select(np.empty((0, 0)), target)
    second = ClassificationAdaptiveAnchorSelector(
        config).select(np.empty((0, 0)), target)

    np.testing.assert_array_equal(first, second)
    assert set(target[first]) == {10, 20, 30}


def test_classification_pair_targets_support_labels_not_starting_at_zero():
    features = np.array([[0.0], [1.0], [2.0]])
    target = np.array([10, 20, 10])
    config = PairwiseLearningConfig(backend="numpy", max_pairs=20)

    batch = build_pair_batch(features, target, np.array(
        [0, 1, 2]), config, task="classification")

    np.testing.assert_array_equal(
        batch.target,
        np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]),
    )
    assert batch.features.shape == (9, 3)
    assert batch.diagnostics["n_pairs"] == 9


def test_classification_pair_target_semantics_same_is_zero_current_contract():
    features = np.array([[0.0], [1.0], [2.0]])
    target = np.array([5, 5, 7])
    config = PairwiseLearningConfig(backend="numpy", max_pairs=20)

    batch = build_pair_batch(features, target, np.array(
        [0]), config, task="classification")

    np.testing.assert_array_equal(batch.target, np.array([0, 0, 1]))


def test_pair_target_semantics_is_reported_in_diagnostics():
    features = np.array([[0.0], [1.0]])
    config = PairwiseLearningConfig(backend="numpy", max_pairs=20)

    clf_batch = build_pair_batch(features, np.array(
        [0, 1]), np.array([0, 1]), config, task="classification")
    clf_semantics = clf_batch.diagnostics["pair_target_semantics"]
    assert clf_semantics["same_label"] == 0
    assert clf_semantics["different_label"] == 1
    assert clf_semantics["target_type"] == "dissimilarity"

    reg_batch = build_pair_batch(features, np.array(
        [1.0, 3.0]), np.array([0, 1]), config, task="regression")
    reg_semantics = reg_batch.diagnostics["pair_target_semantics"]
    assert reg_semantics["delta_sign"] == "left_minus_anchor"
    assert reg_semantics["inference_reconstruction"] == "anchor_target + predicted_delta"


class _PairProbaStub:
    def __init__(self, probabilities: np.ndarray, classes: np.ndarray):
        self.classes_ = classes
        self._probabilities = probabilities

    def predict_proba(self, pair_features: np.ndarray) -> np.ndarray:
        return self._probabilities


def test_predict_same_probability_uses_same_label_column():
    stub_model = _PairProbaStub(
        probabilities=np.array([[0.9, 0.1], [0.2, 0.8]]),
        classes=np.array([0, 1]),
    )

    same_probability = _predict_same_probability(stub_model, np.zeros((2, 1)))

    np.testing.assert_allclose(same_probability, np.array([0.9, 0.2]))


def test_regression_pair_target_uses_left_minus_anchor_sign_convention():
    features = np.array([[0.0], [1.0]])
    target = np.array([1.0, 3.0])
    config = PairwiseLearningConfig(backend="numpy", max_pairs=20)

    batch = build_pair_batch(features, target, np.array(
        [0, 1]), config, task="regression")

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


def test_backward_compatible_pairwise_core_facades_match_strategy_batch():
    features = np.array([[0.0], [1.0]])
    config = PairwiseLearningConfig(backend="numpy", max_pairs=20)

    clf_batch = build_classification_pairs(
        features, np.array([0, 1]), np.array([0, 1]), config)
    reg_batch = build_regression_pairs(
        features, np.array([1.0, 3.0]), np.array([0, 1]), config)

    np.testing.assert_array_equal(clf_batch.target, np.array([0, 1, 1, 0]))
    np.testing.assert_allclose(
        reg_batch.target, np.array([0.0, -2.0, 2.0, 0.0]))
    assert pair_feature_dim(1, "concat_diff") == 3


def test_backward_compatible_anchor_and_aggregation_helpers():
    config = PairwiseLearningConfig(max_pairs=4, anchors_per_class=1)

    clf_anchors = select_classification_anchor_indices(
        np.array([0, 0, 1, 1]), config)
    reg_anchors = select_regression_anchor_indices(
        np.array([0.0, 1.0, 2.0, 3.0]), config)
    proba = aggregate_similarity_to_class_proba(
        np.array([[0.9, 0.2], [0.1, 0.8]]),
        np.array([0, 1]),
        n_classes=2,
    )

    assert set(clf_anchors.tolist()) == {0, 2}
    np.testing.assert_array_equal(reg_anchors, np.array([0]))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(2))


def test_normalize_feature_matrix_handles_scalar_vector_tensor_and_non_finite():
    np.testing.assert_allclose(
        normalize_feature_matrix(3.0), np.array([[3.0]]))
    np.testing.assert_allclose(normalize_feature_matrix(
        [1.0, 2.0]), np.array([[1.0], [2.0]]))

    tensor_like = np.array([[[1.0, np.nan], [np.inf, -np.inf]]])
    normalized = normalize_feature_matrix(tensor_like)

    assert normalized.shape == (1, 4)
    np.testing.assert_allclose(normalized, np.array([[1.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("concat_absdiff", [[1.0, 3.0, 4.0, 1.0, 3.0, 2.0]]),
        ("diff_only", [[-3.0, 2.0]]),
    ],
)
def test_build_pair_features_covers_non_default_modes(mode, expected):
    pairs = build_pair_features(
        np.array([[1.0, 3.0]]),
        np.array([[4.0, 1.0]]),
        PairwiseLearningConfig(backend="numpy", pair_feature_mode=mode),
    )

    np.testing.assert_allclose(pairs, np.asarray(expected, dtype=np.float32))


def test_pair_feature_helpers_report_invalid_inputs():
    with pytest.raises(ValueError, match="Unsupported pair feature mode"):
        pair_feature_dim(3, "unknown")

    with pytest.raises(ValueError, match="equal feature counts"):
        build_pair_features(
            np.ones((2, 2)),
            np.ones((1, 3)),
            PairwiseLearningConfig(backend="numpy"),
        )


def test_resolve_pairwise_backend_reports_unavailable_torch(monkeypatch):
    monkeypatch.setattr(pair_features_module, "torch", None)

    assert resolve_pairwise_backend("auto") == ("numpy", None)
    with pytest.raises(RuntimeError, match="torch is unavailable"):
        resolve_pairwise_backend("cuda")
    with pytest.raises(ValueError, match="Unsupported PDL backend"):
        resolve_pairwise_backend("unknown")


def test_resolve_pairwise_backend_reports_unavailable_cuda(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

        @staticmethod
        def device(name):
            return f"device:{name}"

    monkeypatch.setattr(pair_features_module, "torch", _FakeTorch)

    assert resolve_pairwise_backend("cpu") == ("torch_cpu", "device:cpu")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_pairwise_backend("cuda")


def test_torch_pair_feature_builder_modes_with_fake_torch(monkeypatch):
    class _FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        @property
        def shape(self):
            return self.values.shape

        def repeat_interleave(self, repeats, dim=0):
            return _FakeTensor(np.repeat(self.values, repeats, axis=dim))

        def repeat(self, repeats, axis_repeats):
            return _FakeTensor(np.tile(self.values, (repeats, axis_repeats)))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def __sub__(self, other):
            return _FakeTensor(self.values - other.values)

    class _FakeTorch:
        float32 = np.float32

        @staticmethod
        def as_tensor(values, dtype=None, device=None):
            del dtype, device
            return _FakeTensor(values)

        @staticmethod
        def cat(tensors, dim=1):
            return _FakeTensor(np.concatenate([tensor.values for tensor in tensors], axis=dim))

        @staticmethod
        def abs(tensor):
            return _FakeTensor(np.abs(tensor.values))

    monkeypatch.setattr(pair_features_module, "torch", _FakeTorch)

    diff = _build_pair_features_torch(
        np.array([[1.0, 3.0]]),
        np.array([[4.0, 1.0]]),
        "diff_only",
        device=None,
    )
    absdiff = _build_pair_features_torch(
        np.array([[1.0, 3.0]]),
        np.array([[4.0, 1.0]]),
        "concat_absdiff",
        device=None,
    )

    np.testing.assert_allclose(diff, np.array([[-3.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(
        absdiff,
        np.array([[1.0, 3.0, 4.0, 1.0, 3.0, 2.0]], dtype=np.float32),
    )


def test_pair_target_semantics_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unsupported PDL task"):
        pair_target_semantics(task="forecasting")

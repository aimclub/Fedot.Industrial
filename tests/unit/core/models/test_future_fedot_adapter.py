"""Focused public-boundary tests for FUTURE packed-array adapters."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from fedot_ind.core.models.future.fedot_adapter import ModalitySlice, PackedFusionClassifier
from fedot_ind.core.models.future.tools import FutureTrainingConfig
from fedot_ind.core.models.future.trainer import FutureClassifierTrainer
from fedot_ind.core.multimodal.enums import MultimodalModality


LAYOUT = [{"name": "raw", "shape": [1, 16]}, {"name": "stats", "shape": [3]}]


def make_classifier(**overrides):
    options = dict(
        modality_layout=LAYOUT,
        d_model=4,
        encoder_kwargs={"raw": {"hidden_channels": [4], "dropout": 0.0},
                        "stats": {"hidden_dims": [4], "dropout": 0.0}},
        training_config={"epochs": 1, "batch_size": 4, "device": "cpu", "seed": 19,
                         "timing_warmup": False},
    )
    options.update(overrides)
    return PackedFusionClassifier(**options)


@pytest.fixture
def training_data():
    X = np.random.default_rng(7).normal(size=(8, 19)).astype(np.float32)
    y = np.array([-7, 11] * 4)
    return X, y


@pytest.mark.parametrize("tensor_input", [False, True])
def test_selector_returns_exact_independent_columns(tensor_input):
    X = np.arange(38, dtype=np.float64).reshape(2, 19)
    values = torch.from_numpy(X.copy()) if tensor_input else X.copy()
    selector = ModalitySlice(input_width=19, start=16, stop=19)
    assert selector.fit(values) is selector
    transformed = selector.transform(values)
    assert isinstance(transformed, type(values))
    np.testing.assert_array_equal(transformed, X[:, 16:19])
    transformed[:] = -1
    np.testing.assert_array_equal(values, X)


@pytest.mark.parametrize("bounds", [(0, 0, 1), (19, -1, 4), (19, 4, 4), (19, 7, 6),
                                    (19, 0, 20), (True, 0, 1), (19, 0.0, 3)])
def test_selector_validates_bounds_early(bounds):
    with pytest.raises((ValueError, TypeError)):
        ModalitySlice(*bounds)


@pytest.mark.parametrize("method", ["fit", "transform"])
def test_selector_checks_exact_input_width(method):
    selector = ModalitySlice(19, 0, 16)
    with pytest.raises(ValueError, match="width 19"):
        getattr(selector, method)(np.zeros((2, 20)))


@pytest.mark.parametrize("fusion_method", ["concat", "gated", "film", "raw_centered_residual",
                                           "ordinary_bottleneck", "raw_residual_bottleneck",
                                           "context_only_residual_bottleneck"])
def test_actual_training_reuses_trainer_and_preserves_domain_labels(training_data, fusion_method):
    X, y = training_data
    X_before, y_before = X.copy(), y.copy()
    threads_before = torch.get_num_threads()
    classifier = make_classifier(fusion_method=fusion_method)
    assert classifier.fit(X, y.reshape(-1, 1)) is classifier
    assert isinstance(classifier.trainer_, FutureClassifierTrainer)
    assert classifier.model_ is classifier.trainer_.model
    assert classifier.model_.modalities == (MultimodalModality.raw, MultimodalModality.stats)
    assert classifier.trainer_._build_shapes == {
        MultimodalModality.raw: (8, 1, 16), MultimodalModality.stats: (8, 3),
    }
    assert len(classifier.trainer_.history.train_loss) == 1
    np.testing.assert_array_equal(classifier.classes_, [-7, 11])
    probabilities = classifier.predict_proba(X[:3])
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
    predictions = classifier.predict(X[:3])
    np.testing.assert_array_equal(predictions, classifier.classes_[probabilities.argmax(axis=1)])
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    assert torch.get_num_threads() == threads_before


def test_fit_is_deterministic_including_weight_initialization(training_data):
    X, y = training_data
    first = make_classifier().fit(X, y)
    torch.rand(40)
    second = make_classifier().fit(torch.from_numpy(X), torch.from_numpy(y))
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
    for key, weights in first.model_.state_dict().items():
        assert torch.equal(weights, second.model_.state_dict()[key])


def test_numeric_nonintegral_class_labels_are_not_truncated(training_data):
    X, _ = training_data
    y = np.array([-1.5, 8.75] * 4)
    classifier = make_classifier().fit(X, y)
    np.testing.assert_array_equal(classifier.classes_, [-1.5, 8.75])
    assert set(classifier.predict(X)).issubset(set(y))


def test_bfloat16_tensor_targets_keep_numeric_labels(training_data):
    X, _ = training_data
    y = torch.tensor([-1.5, 8.75] * 4, dtype=torch.bfloat16)
    classifier = make_classifier().fit(X, y)
    np.testing.assert_array_equal(classifier.classes_, [-1.5, 8.75])


def test_training_bundle_does_not_alias_caller_inputs(monkeypatch, training_data):
    X, y = training_data
    original_X, original_y = X.copy(), y.copy()

    def mutate_bundle_then_fail(self, loader, *, build_bundle):
        for tensor in build_bundle.modalities.values():
            tensor.fill_(100)
        build_bundle.target.fill_(0)
        raise RuntimeError("mutation probe")

    monkeypatch.setattr(FutureClassifierTrainer, "fit", mutate_bundle_then_fail)
    with pytest.raises(RuntimeError, match="mutation probe"):
        make_classifier().fit(torch.from_numpy(X), torch.from_numpy(y))
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_predict_before_fit_is_rejected(method, training_data):
    X, _ = training_data
    with pytest.raises(ValueError, match="not fitted"):
        getattr(make_classifier(), method)(X)


@pytest.mark.parametrize("bad_y,match", [
    (np.zeros(8), "at least two unique"),
    (np.arange(7), "one class label per sample"),
    (np.zeros((8, 2)), "one class label per sample"),
    (np.array([0, np.nan] * 4), "finite"),
    (np.array([0, np.inf] * 4), "finite"),
    (np.array([False, True] * 4), "numeric"),
    (np.array(["a", "b"] * 4), "numeric"),
])
def test_invalid_training_targets_leave_no_fitted_state(training_data, bad_y, match):
    X, _ = training_data
    classifier = make_classifier()
    with pytest.raises((TypeError, ValueError), match=match):
        classifier.fit(X, bad_y)
    assert classifier.model_ is classifier.trainer_ is classifier.classes_ is None
    with pytest.raises(ValueError, match="not fitted"):
        classifier.predict(X)


def test_failed_refit_discards_previous_fitted_state(training_data):
    X, y = training_data
    classifier = make_classifier().fit(X, y)
    with pytest.raises(ValueError, match="width"):
        classifier.fit(X[:, :-1], y)
    assert classifier.model_ is classifier.trainer_ is classifier.classes_ is None
    with pytest.raises(ValueError, match="not fitted"):
        classifier.predict_proba(X)


def test_training_exception_does_not_publish_partial_state(monkeypatch, training_data):
    X, y = training_data

    def fail(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(FutureClassifierTrainer, "fit", fail)
    classifier = make_classifier()
    with pytest.raises(RuntimeError, match="training failed"):
        classifier.fit(X, y)
    assert classifier.model_ is classifier.trainer_ is classifier.classes_ is None


@pytest.mark.parametrize("bad_X,match", [
    (np.zeros((2, 20)), "width"),
    (np.zeros((2, 1, 19)), "2-D"),
    (np.zeros((0, 19)), "empty batches"),
    (np.full((2, 19), np.nan), "finite"),
])
def test_prediction_validates_exact_shape_and_finiteness(training_data, bad_X, match):
    X, y = training_data
    classifier = make_classifier().fit(X, y)
    with pytest.raises(ValueError, match=match):
        classifier.predict_proba(bad_X)
    assert classifier.predict(X[:1]).shape == (1,)


@pytest.mark.parametrize("options", [
    {"d_model": True}, {"d_model": 0}, {"d_model": 2.5},
    {"head_dropout": float("nan")}, {"head_dropout": 1.1}, {"head_hidden_dim": False},
    {"fusion_method": "not-a-fusion"}, {"fusion_kwargs": {"unknown": 1}},
    {"fusion_kwargs": {"dropout": float("nan")}}, {"fusion_kwargs": {"hidden_dim": True}},
    {"fusion_kwargs": {"activation": "unknown"}},
    {"fusion_method": "ordinary_bottleneck", "d_model": 3},
    {"fusion_method": "ordinary_bottleneck", "fusion_kwargs": {"num_heads": 0}},
    {"fusion_method": "ordinary_bottleneck", "fusion_kwargs": {"pooling": "invalid"}},
    {"encoder_kwargs": {"unknown": {}}}, {"encoder_kwargs": {"raw": []}},
    {"encoder_kwargs": {"raw": None}},
    {"encoder_kwargs": {"raw": {"d_model": 4}}},
    {"encoder_kwargs": {"stats": {"hidden_dims": [True]}}},
    {"training_config": {"epochs": 0}}, {"training_config": {"batch_size": True}},
    {"training_config": {"seed": True}}, {"training_config": {"seed": 2**64}},
    {"training_config": {"learning_rate": 0}}, {"training_config": {"weight_decay": float("inf")}},
    {"training_config": {"unknown": 1}}, {"training_config": {"device": None}},
    {"training_config": {"timing_warmup": "false"}},
])
def test_invalid_configuration_is_rejected_before_fit(options):
    with pytest.raises((TypeError, ValueError)):
        make_classifier(**options)


def test_raw_centered_fusion_rejects_no_context_early():
    with pytest.raises(ValueError, match="context modality"):
        PackedFusionClassifier([LAYOUT[0]], fusion_method="film")


def test_configuration_is_copied_and_default_training_values_are_reused():
    layout = deepcopy(LAYOUT)
    encoder_kwargs = {"raw": {"hidden_channels": [4]}}
    fusion_kwargs = {"hidden_dim": 8}
    config = FutureTrainingConfig(epochs=1)
    classifier = PackedFusionClassifier(layout, encoder_kwargs=encoder_kwargs,
                                        fusion_kwargs=fusion_kwargs, training_config=config)
    assert encoder_kwargs == {"raw": {"hidden_channels": [4]}}
    layout[0]["shape"][1] = 99
    encoder_kwargs["raw"]["hidden_channels"][0] = 99
    fusion_kwargs["hidden_dim"] = 99
    config.epochs = 99
    assert classifier.layout[0].shape == (1, 16)
    assert classifier.encoder_kwargs["raw"]["hidden_channels"] == [4]
    assert classifier.fusion_kwargs["hidden_dim"] == 8
    assert classifier.training_config.epochs == 1
    default = PackedFusionClassifier(LAYOUT).training_config
    assert default.epochs == FutureTrainingConfig().epochs
    assert default.batch_size == FutureTrainingConfig().batch_size

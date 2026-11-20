import numpy as np
import pytest

from fedot_ind.core.models.early_classification import PrefixSchedule, evaluate_decisions, run_early
from fedot_ind.core.models.early_classification.confidence import (
    _calibration_splits, _features, fit_ecec, fit_prefix_probabilities, fit_teaser,
)
from fedot_ind.core.models.early_classification.economy import fit_cluster_economy, fit_gamma_economy
from fedot_ind.core.models.early_classification.neighbors import _stable_lengths, fit_early_nearest_neighbor
from fedot_ind.core.models.early_classification.shapelets import fit_early_shapelets


def panel():
    labels = np.repeat([0, 1], 8)
    x = np.random.default_rng(42).normal(0, .02, size=(16, 2, 8)) + 2 * labels[:, None, None]
    return x, labels, PrefixSchedule((2, 4, 8), 2, (1, 0))


def test_prefix_normalization_ignores_translation_and_handles_constants():
    x = np.array([[[1., 5., 6.], [2., 2., 2.]]])
    transformed = _features(x, True)
    np.testing.assert_allclose(transformed, _features(3 * x + 80, True))
    np.testing.assert_array_equal(transformed[0, 3:], 0.)


def test_grouped_probability_calibration_keeps_repeated_records_together():
    x, y, schedule = panel()
    groups = np.repeat(np.arange(8), 2)
    splits = _calibration_splits(x, y, schedule.classes, 3, 10, groups)
    for fit, holdout in splits:
        assert set(groups[fit]).isdisjoint(groups[holdout])
    np.testing.assert_array_equal(np.sort(np.concatenate([held for _, held in splits])), np.arange(len(x)))
    model = fit_prefix_probabilities(x, y, schedule, groups=groups, seed=10)
    np.testing.assert_allclose(model.out_of_fold.sum(axis=-1), 1.)
    with pytest.raises(ValueError, match="group-disjoint"):
        fit_prefix_probabilities(x, y, schedule, groups=y, seed=10)


def test_prefix_causality_and_forced_final_decision():
    x, y, schedule = panel()
    decision = run_early(x[0], schedule, lambda prefix, step: int(prefix.mean() > 1), lambda *_: 1)
    changed = x[0].copy()
    changed[:, 2:] = np.nan
    assert decision == run_early(changed, schedule, lambda prefix, step: int(prefix.mean() > 1), lambda *_: 1)
    forced = run_early(x[0], schedule, lambda *_: None, lambda *_: 0)
    assert forced.observed == 8 and forced.forced
    assert evaluate_decisions([decision, forced], [y[0], y[0]], [8, 8]) == {
        "accuracy": 1., "observed_fraction": .625, "delay_cost": None,
    }


def test_calibration_and_cost_policies():
    x, y, schedule = panel()
    trained = fit_prefix_probabilities(x, y, schedule, folds=2)
    scores = trained.out_of_fold
    assert np.array_equal(scores[:, -1].argmax(axis=-1) == 0, y == 1)
    gate = fit_teaser(scores, y, schedule.classes)
    assert gate.choose(scores[0, :1], x[0, :, :2]) is None
    assert gate.choose(np.array([[.9, .1], [.1, .9]]), x[0, :, :4]) is None
    ecec = fit_ecec(scores, y, schedule.classes, threshold=.8)
    assert trained.predict_one(x[0], ecec).label == 0
    cost, delay = [[0, 1], [2, 0]], [0, 3, 6]
    for mode in ("k", "multi_k"):
        model = fit_cluster_economy(x, y, schedule, scores, cost, delay, groups=2, mode=mode)
        assert trained.predict_one(x[0], model).observed == 2
    for mode in ("gamma", "lite"):
        model = fit_gamma_economy(schedule, scores, y, cost, delay, groups=2, mode=mode)
        assert trained.predict_one(x[-1], model).observed == 2
        for transition in model.transitions:
            np.testing.assert_allclose(transition.sum(axis=1), 1.)


def test_shapelet_votes_and_reverse_neighbor_minima():
    x, y, schedule = panel()
    model = fit_early_shapelets(x, y, schedule, widths=(2, 3), min_channels=2, candidates_per_channel=6)
    assert set(item.channel for item in model.templates) == {0, 1}
    assert model.predict_one(x[-1]).label == 1
    one_channel = model.predict_one(np.vstack((x[-1, 0], np.full(8, 100.))))
    assert one_channel.forced
    nearest = fit_early_nearest_neighbor(x, y, schedule)
    assert nearest.predict_one(x[0]).label == 0
    np.testing.assert_array_equal(
        _stable_lengths(np.array([[1, 0, 0], [1, 2, 1], [1, 2, 1]]), np.array([0, 0, 1]), []),
        [1, 1, 1],
    )


def test_mlstm_uses_both_branches_and_ignores_unobserved_suffix():
    torch = pytest.importorskip("torch")
    from fedot_ind.core.models.early_classification.mlstm_fcn import fit_mlstm_fcn

    x, y, schedule = panel()
    model = fit_mlstm_fcn(x, y, schedule, cutoff=4, width=4, epochs=1)
    changed = x.copy()
    changed[:, :, 4:] = np.nan
    np.testing.assert_array_equal(model.predict_proba(x), model.predict_proba(changed))
    model.network(torch.ones(2, 2, 4)).sum().backward()
    assert model.network.recurrent.weight_ih_l0.grad is not None
    assert model.network.convolution[0].weight.grad is not None


def test_fixed_prefix_training_uses_only_the_selected_observations():
    pytest.importorskip("torch")
    from fedot_ind.core.models.early_classification.mlstm_fcn import fit_mlstm_fcn

    x, y, schedule = panel()

    def fit(features):
        return fit_mlstm_fcn(features, y, schedule, cutoff=4, width=4, epochs=1)

    expected = fit(x).predict_proba(x)
    unavailable = x.copy()
    unavailable[:, :, 4:] = np.nan
    for features in (unavailable, x[:, :, :4]):
        np.testing.assert_array_equal(fit(features).predict_proba(x), expected)
    unavailable[0, 0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit(unavailable)

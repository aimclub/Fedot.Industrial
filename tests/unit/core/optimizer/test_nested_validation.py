"""Nested TRAIN split laws, inner-only decisions, and locked outer evaluation."""

from copy import deepcopy
from dataclasses import asdict, FrozenInstanceError
import json
from pathlib import Path

from hypothesis import given, strategies as st
import numpy as np
import pytest

from benchmark.industrial.experiments.mutation_planning.population_study.nested_validation import core, runner


def settings():
    return runner.load_settings()[:2]


def test_frozen_configuration_and_approved_fit_counts(tmp_path, monkeypatch):
    config, grid = settings()
    assert (config.outer_folds, config.inner_folds, config.outer_seed, config.inner_seed) == (5, 3, 42, 123)
    with pytest.raises(FrozenInstanceError):
        config.outer_seed = 77
    monkeypatch.setattr(runner, '_train_arrays', lambda *args: pytest.fail('No dataset reads while planning'))
    protocol = Path(__file__).resolve().parents[4] / 'docs/dev_guide/evo04_population_config_20261019.json'
    result = runner.run(protocol, tmp_path / 'absent')
    assert result['inner_grid_fits'] == 360
    assert result['total_ridge_fits'] == 455
    assert result['pdl_fits'] == 5
    assert len(core.candidates(grid)) == 24
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('update', [
    {'outer_folds': True}, {'inner_folds': 1}, {'outer_seed': -1}, {'feature_seed': None},
    {'inner_seed': 1.5}, {'tie_break': 'outer_loss'}, {'ridge_grid_sha256': 'bad'}, {'unknown': 1},
])
def test_config_rejects_ambiguous_fields(update):
    raw = runner.load_settings()[2]
    with pytest.raises(ValueError):
        core.parse_config({**raw, **update})


def test_cli_cannot_override_agreed_fold_count_or_grid(tmp_path, monkeypatch):
    config, grid, raw, grid_raw = runner.load_settings()
    override = tmp_path / 'config.json'
    override.write_text(json.dumps({**raw, 'outer_folds': 4}))
    with pytest.raises(ValueError, match='approved bundled protocol'):
        runner.load_settings(override)

    class Resource:
        def joinpath(self, name):
            return self

        def read_text(self, **kwargs):
            return json.dumps({**grid_raw, 'alphas': [.1, 1, 10]})

    original = runner.resources.files
    monkeypatch.setattr(runner.resources, 'files', lambda name: Resource() if name is runner.ridge_diagnostics
                        else original(name))
    with pytest.raises(ValueError, match='Ridge grid differs'):
        runner.load_settings()


def test_full_protocol_pins_the_pdl_baseline_without_reading_data(tmp_path):
    path = Path(__file__).resolve().parents[4] / 'docs/dev_guide/evo04_population_config_20261019.json'
    protocol = json.loads(path.read_text())
    protocol['operation_parameters']['pdl_reg']['max_pairs'] = 1024
    changed = tmp_path / 'changed.json'
    changed.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='Population protocol differs'):
        runner.run(changed, tmp_path / 'absent')
    assert not (tmp_path / 'absent').exists()


@given(samples=st.integers(min_value=25, max_value=100))
def test_nested_partition_conservation_determinism_and_disjointness(samples):
    config, _ = settings()
    plan = core.make_plan(samples, config)
    assert plan == core.make_plan(samples, config)
    assert sorted(index for fold in plan for index in fold.split.validation) == list(range(samples))
    for fold in plan:
        tr, val = set(fold.split.training), set(fold.split.validation)
        assert not tr & val and tr | val == set(range(samples))
        assert sorted(index for split in fold.inner for index in split.validation) == list(range(len(tr)))
        for inner in fold.inner:
            assert not set(inner.training) & set(inner.validation)
            mapped = {fold.split.training[index] for index in inner.training + inner.validation}
            assert mapped == tr and not mapped & val


def test_exact_naturalgas_fold_sizes():
    config, _ = settings()
    for fold in core.make_plan(65, config):
        assert len(fold.split.training) == 52 and len(fold.split.validation) == 13
        assert [(len(s.training), len(s.validation)) for s in fold.inner] == [(34, 18), (35, 17), (35, 17)]


def inner_scores(grid, loss=lambda fold, candidate: 1.):
    return tuple(core.InnerScore(fold, candidate, loss(fold, candidate)) for fold in range(3)
                 for candidate in core.candidates(grid))


def test_exact_ties_prefer_simple_none_then_larger_alpha_independent_of_row_order():
    _, grid = settings()
    rows = inner_scores(grid)
    expected = core.Candidate('simple', 'none', 1000.)
    assert core.select_candidate(rows, grid, 3).candidate == expected
    assert core.select_candidate(tuple(reversed(rows)), grid, 3).candidate == expected
    choice = core.Candidate('composite', 'standard', .01)
    rows = inner_scores(grid, lambda fold, candidate: np.nextafter(1., 0.).item() if candidate == choice else 1.)
    assert core.select_candidate(rows, grid, 3).candidate == choice


def test_selection_uses_unweighted_mean_of_inner_normalized_losses():
    _, grid = settings()
    choice = core.candidates(grid)[0]
    rows = inner_scores(grid, lambda fold, candidate: [.1, .2, .6][fold] if candidate == choice else .4)
    selection = core.select_candidate(rows, grid, 3)
    assert selection.candidate == choice
    assert selection.mean_inner_loss == pytest.approx(.3)
    assert selection.margin_to_second == pytest.approx(.1)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'nan', 'inf', 'negative', 'bool'])
def test_selection_rejects_incomplete_or_failed_inner_evidence(damage):
    _, grid = settings()
    rows = list(inner_scores(grid))
    if damage == 'missing':
        rows.pop()
    elif damage == 'duplicate':
        rows.append(rows[0])
    else:
        value = {'nan': float('nan'), 'inf': float('inf'), 'negative': -1., 'bool': True}[damage]
        rows[0] = core.InnerScore(rows[0].fold, rows[0].candidate, value)
    with pytest.raises(ValueError):
        core.select_candidate(tuple(rows), grid, 3)


def outer_records(plan):
    y = np.linspace(-.5, .5, sum(len(fold.split.validation) for fold in plan))
    rows = []
    for fold in plan:
        for offset, method in enumerate(('selected', 'median', 'simple_alpha1', 'composite_alpha1', 'q_pdl'), start=1):
            target = y[list(fold.split.validation)]
            prediction = target + offset * .01
            scale = float(np.std(y[list(fold.split.training)]))
            rows.append({'outer_fold': fold.ordinal, 'method': method,
                         'training_indices': list(fold.split.training),
                         'validation_indices': list(fold.split.validation),
                         'target': target.tolist(), 'prediction': prediction.tolist(), 'normalization_scale': scale,
                         'outer_loss': offset * .01 / scale})
    return rows


def test_outer_summary_replays_metrics_and_covers_each_object_once_per_method():
    config, _ = settings()
    plan = core.make_plan(65, config)
    rows = outer_records(plan)
    result = core.summarize_outer(rows, plan)
    assert all(row['objects'] == 65 for row in result['methods'])
    assert result['methods'][0]['oof_rmse_original_units'] == pytest.approx(.01)
    assert all(row['selected_better_folds'] == 5 for row in result['methods'][1:])
    assert not result['full_study_accepted'] and not result['official_test_used']
    assert core.summarize_outer(list(reversed(rows)), plan) == result


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'target', 'train_idx', 'validation_idx', 'scale', 'nan'])
def test_outer_summary_rejects_conflicting_or_partial_evidence(damage):
    config, _ = settings()
    plan = core.make_plan(65, config)
    rows = outer_records(plan)
    if damage == 'missing':
        rows.pop()
    elif damage == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif damage == 'target':
        rows[0]['target'][0] += 1.
    elif damage == 'train_idx':
        rows[0]['training_indices'].reverse()
    elif damage == 'validation_idx':
        rows[0]['validation_indices'].reverse()
    elif damage == 'scale':
        for row in rows:
            row['normalization_scale'] *= 2
            row['outer_loss'] /= 2
    else:
        rows[0]['outer_loss'] = float('nan')
    with pytest.raises(ValueError):
        core.summarize_outer(rows, plan)


def test_outer_labels_cannot_change_inner_scores_scalers_or_choice_and_lock_precedes_evaluation(tmp_path, monkeypatch):
    config, grid = settings()
    fold = core.make_plan(65, config)[0]
    x = np.column_stack((np.arange(65, dtype=float), np.sin(np.arange(65))))
    y = np.linspace(-.5, .5, 65)
    modified = y.copy()
    modified[list(fold.split.validation)] += 1000
    seen_training = []

    def extract(study, structure, seed, features, target, training, validation, grid):
        seen_training.append(tuple(features[training, 0].astype(int)))
        return features[training].copy(), features[validation].copy(), target[training].copy(), {}

    def pdl(study, config, features, target, training, validation):
        assert set(training) == set(fold.split.training)
        return np.full(len(validation), target[training].mean()), {'training_sample_count': len(training)}

    original_outer = runner.evaluate_outer

    def outer(*args):
        directory = args[-1]
        locked = json.loads((directory / 'selection.json').read_text())
        assert not locked['outer_evaluated']
        assert locked['candidate'] == asdict(args[-2].candidate)
        return original_outer(*args)

    monkeypatch.setattr(runner, 'extract_features', extract)
    monkeypatch.setattr(runner, 'predict_pdl', pdl)
    monkeypatch.setattr(runner, 'evaluate_outer', outer)
    results = []
    for index, target in enumerate((y, modified)):
        directory = tmp_path / str(index)
        directory.mkdir()
        results.append(runner.run_fold({}, config, grid, x, target, fold, directory))
    a, b = (json.loads((tmp_path / str(index) / 'inner_records.json').read_text()) for index in range(2))
    for original, changed in zip(a, b):
        for key in ('validation_loss', 'scaler', 'original_coefficient', 'validation_prediction'):
            assert original[key] == changed[key]
    assert results[0][1]['selected'] == results[1][1]['selected']
    assert results[0][0][0]['outer_loss'] != results[1][0][0]['outer_loss']
    assert all(set(indices) <= set(fold.split.training) for indices in seen_training)
    np.testing.assert_array_equal(y, np.linspace(-.5, .5, 65))


@pytest.mark.parametrize('target', [[1., 1.], [float('nan'), 2.], [float('inf'), 3.]])
def test_invalid_training_scale_is_not_silently_skipped(target):
    with pytest.raises(ValueError, match='positive standard deviation'):
        runner.positive_scale(target)


def test_native_pdl_keeps_anchors_and_base_seed_on_training_subset_only():
    config, _ = settings()
    path = Path(__file__).resolve().parents[4] / 'docs/dev_guide/evo04_population_config_20261019.json'
    study = runner.load_protocol(path).to_record()
    t = np.linspace(0, 6, 32)
    x = np.asarray([np.sin(t + index / 5) + (index + 1) / 8 for index in range(16)])
    y = np.linspace(-.5, .5, 16)
    changed = y.copy()
    changed[12:] += 1000
    training, validation = np.arange(12), np.arange(12, 16)
    with runner._runtime():
        prediction, evidence = runner.predict_pdl(study, config, x, y, training, validation)
        modified_prediction, modified_evidence = runner.predict_pdl(
            study, config, x, changed, training, validation)
    np.testing.assert_array_equal(prediction, modified_prediction)
    assert evidence['training_sample_count'] == 12
    assert all(0 <= index < 12 for index in evidence['anchor_indices'])
    assert evidence['pdl_diagnostics'] == modified_evidence['pdl_diagnostics']
    assert evidence['pdl_diagnostics']['config']['max_pairs'] == 512
    assert evidence['pdl_diagnostics']['resolved_base_model_params']['random_state'] == 42

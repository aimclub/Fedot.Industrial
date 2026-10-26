"""Frozen DAGs, exact ranks, TRAIN isolation and the 100-fit budget."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from hypothesis import given, strategies as st
import numpy as np
import pytest

from benchmark.industrial.experiments.mutation_planning.population_study.winner_ranking import core, runner


ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / 'docs/dev_guide/evo04_candidate_pool_20261020.json'
PROTOCOL = ROOT / 'docs/dev_guide/evo04_population_config_20261019.json'


def settings():
    return runner.load_settings(AUDIT)


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
def test_frozen_audit_accepts_git_line_endings_and_rejects_content_drift(tmp_path, newline):
    original = AUDIT.read_bytes().replace(b'\r\n', b'\n')
    path = tmp_path / 'audit.json'
    path.write_bytes(original.replace(b'\n', newline))
    config, _, candidates = runner.load_settings(path)
    assert {candidate.name for candidate in candidates} == set(config['candidates'])
    changed = original.replace(b'"schema_version"', b'"schema_versioN"', 1)
    assert changed != original
    path.write_bytes(changed.replace(b'\n', newline))
    with pytest.raises(ValueError, match='frozen TRAIN-only'):
        runner.load_settings(path)


def test_dry_run_does_not_read_dataset_or_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_train_arrays', lambda *args: pytest.fail('No dataset access in preview'))
    monkeypatch.setattr(runner, 'build_pipeline', lambda *args: pytest.fail('No model construction in preview'))
    preview = runner.run(PROTOCOL, AUDIT, tmp_path / 'output')
    assert (preview['inner_fits'], preview['outer_fits'], preview['total_native_pipeline_fits']) == (75, 25, 100)
    assert not (tmp_path / 'output').exists()
    assert not preview['official_test_read_or_hashed'] and not preview['full_study_accepted']


def test_all_native_dags_round_trip_without_fitting_and_c03_shares_its_parent():
    _, _, choices = settings()
    with runner._runtime():
        for choice in choices:
            pipeline = runner.build_pipeline(choice)
            assert len(pipeline.nodes) == len(choice.configuration['nodes'])
            assert all(node.fitted_operation is None for node in pipeline.nodes)
            if choice.name == 'C03':
                qt, q = pipeline.root_node._nodes_from_with_fixed_order()
                assert qt.nodes_from[0] is q
                assert qt.parameters['task_type'] == 'regression'
            if choice.name == 'C32':
                qt = next(node for node in pipeline.nodes if node.operation.operation_type.endswith('_torch'))
                assert 'task_type' not in qt.parameters


@pytest.mark.parametrize('damage', ['parameters', 'duplicate', 'id', 'missing'])
def test_frozen_pool_rejects_config_changes(damage):
    config, _, _ = settings()
    catalogue = json.loads(AUDIT.read_text())['catalogue']
    row = next(row for row in catalogue if row['candidate'] == 'C03')
    if damage == 'parameters':
        row['configuration']['nodes'][0]['parameters']['alpha'] = 10
    elif damage == 'duplicate':
        catalogue.append(deepcopy(row))
    elif damage == 'missing':
        catalogue.remove(row)
    else:
        row['configuration_id'] = 'bad'
    with pytest.raises(ValueError):
        core.admit_candidates(catalogue, config['candidates'])


def score_rows(loss=lambda split, name: 1.):
    names = ('C03', 'C10', 'C16', 'C20', 'C32')
    return [{'inner_fold': split, 'candidate': name, 'loss': loss(split, name)}
            for split in range(3) for name in names]


@given(st.permutations(score_rows()))
def test_exact_tie_selection_is_invariant_to_record_order(rows):
    selection = core.choose_inner(rows, tuple(settings()[0]['candidates']), 3)
    assert selection['candidate'] == 'C03'
    assert [row['average_rank'] for row in selection['ranking']] == [3.] * 5
    assert [row['decision_rank'] for row in selection['ranking']] == [1, 2, 3, 4, 5]


def test_selection_uses_exact_means_and_not_approximate_ties():
    names = tuple(settings()[0]['candidates'])
    rows = score_rows(lambda split, name: np.nextafter(1., 0.).item() if name == 'C32' else 1.)
    assert core.choose_inner(rows, names, 3)['candidate'] == 'C32'
    rows = score_rows(lambda split, name: [.1, .2, .6][split] if name == 'C10' else .4)
    assert core.choose_inner(rows, names, 3)['mean_inner_loss'] == pytest.approx(.3)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'nan', 'inf', 'negative', 'bool'])
def test_incomplete_failed_or_nonfinite_scores_are_not_silently_dropped(damage):
    rows = score_rows()
    if damage == 'missing':
        rows.pop()
    elif damage == 'duplicate':
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]['loss'] = {'nan': float('nan'), 'inf': float('inf'), 'negative': -1., 'bool': True}[damage]
    with pytest.raises(ValueError):
        core.choose_inner(rows, tuple(settings()[0]['candidates']), 3)


def test_spearman_handles_reversal_and_reports_undefined_constant_ranks():
    first = {'C03': 1., 'C10': 2., 'C16': 3.}
    assert core.rank_correlation(first, first) == pytest.approx(1.)
    assert core.rank_correlation(first, {name: 4 - value for name, value in first.items()}) == pytest.approx(-1.)
    assert core.rank_correlation(first, dict.fromkeys(first, 1.)) is None


def test_exact_100_evaluations_locked_selection_and_outer_target_poison(tmp_path, monkeypatch):
    config, nested, choices = settings()
    plan = runner.make_plan(65, nested)
    features = np.column_stack((np.arange(65), np.sin(np.arange(65))))
    target = np.linspace(-.5, .5, 65)
    seen = []

    def evaluate(choice, seed, x, y, xv):
        assert seed == 42
        assert len(x) == len(y)
        if len(y) == 52:
            fold = next(f for f in plan if set(f.split.training) == set(x[:, 0].astype(int)))
            assert (tmp_path / str(fold.ordinal) / 'selection.json').exists()
        seen.append(tuple(x[:, 0].astype(int)))
        return np.full(len(xv), y.mean() + int(choice.name[1:]) / 1000), {'training_samples': len(y)}

    monkeypatch.setattr(runner, 'evaluate_native', evaluate)
    all_inner, all_outer, selections = [], [], {}
    for fold in plan:
        directory = tmp_path / str(fold.ordinal)
        directory.mkdir()
        inner, outer, stage = runner.run_fold(config, choices, features, target, fold, directory)
        all_inner.extend(inner)
        all_outer.extend(outer)
        selections[fold.ordinal] = stage['selection']
    assert len(seen) == 100
    summary = core.summarize(all_inner, all_outer, selections, tuple(c.name for c in choices))
    assert len(summary['rankings']) == 50 and len(summary['paired']) == 10
    assert len(summary['correlations']) == 40
    assert core.summarize(list(reversed(all_inner)), list(reversed(all_outer)), selections,
                          tuple(c.name for c in choices)) == summary

    fold = plan[0]
    modified = target.copy()
    modified[list(fold.split.validation)] += 1000
    other = tmp_path / 'poison'
    other.mkdir()
    # This mocked call checks presence of the original lock as well; no fitted state is reused.
    inner, outer, stage = runner.run_fold(config, choices, features, modified, fold, other)
    original_inner = [row for row in all_inner if row['outer_fold'] == 0]
    assert inner == original_inner
    assert stage['selection'] == selections[0]
    assert outer[0]['loss'] != all_outer[0]['loss']
    assert all(set(indices) <= set(fold.split.training) for indices in seen[-20:-5])
    damaged = deepcopy(all_outer)
    damaged.pop()
    with pytest.raises(ValueError):
        core.summarize(all_inner, damaged, selections, tuple(c.name for c in choices))


def test_guard_blocks_mixed_historical_json_and_test_even_if_paths_do_not_exist(tmp_path):
    with runner.ranking_data_guard('NaturalGasPricesSentiment'):
        for name in ('records.json', 'NaturalGasPricesSentiment_TEST.ts', 'another_TEST.arff'):
            with pytest.raises(RuntimeError, match='forbidden'):
                (tmp_path / name).read_bytes()
    with pytest.raises(FileNotFoundError):
        (tmp_path / 'records.json').read_bytes()


def test_modified_native_configuration_cannot_be_built():
    _, _, choices = settings()
    choice = choices[0]
    raw = choice.configuration
    raw['nodes'][0]['parents'].reverse()
    changed = replace(choice, configuration_json=core.canonical_json(raw))
    with runner._runtime(), pytest.raises(ValueError, match='Native reconstruction changed'):
        runner.build_pipeline(changed)


def synthetic_evidence():
    config, nested, choices = settings()
    plan = runner.make_plan(65, nested)
    from fedot_ind.core.optimizer.randomness import CandidateRandomness
    target = np.linspace(-.5, .5, 65)
    inner, outer = [], []
    for fold in plan:
        for ordinal, split in [*enumerate(fold.inner), (None, fold.split)]:
            mapping = np.asarray(fold.split.training) if ordinal is not None else np.arange(65)
            tr, val = mapping[list(split.training)], mapping[list(split.validation)]
            scale = runner.positive_scale(target[tr])
            for choice in choices:
                row = {'stage': 'inner' if ordinal is not None else 'outer', 'outer_fold': fold.ordinal,
                       'candidate': choice.name, 'status': 'succeeded', 'training': list(split.training),
                       'validation': list(split.validation), 'training_samples': len(tr),
                       'training_target': target[tr].tolist(), 'target': target[val].tolist(),
                       'prediction': (target[val] + .01).tolist(), 'normalization_scale': scale,
                       'loss': .01 / scale, 'fit_seconds': .1, 'predict_seconds': .01,
                       'configuration_id': choice.configuration_id,
                       'evaluation_seed': CandidateRandomness(42, 'evaluation:' + choice.configuration_id, 0).seed}
                if ordinal is not None:
                    row['inner_fold'] = ordinal
                    inner.append(row)
                else:
                    outer.append(row)
    return inner, outer, target, plan, choices


def test_saved_losses_replay_only_with_training_scale_and_local_inner_index_mapping():
    inner, outer, target, plan, choices = synthetic_evidence()
    runner.validate_rows(inner, outer, np.empty((65, 0)), target, plan, choices)


@pytest.mark.parametrize('damage', ['train_target', 'validation_target', 'scale', 'prediction', 'seed', 'identity',
                                    'fit_time', 'status', 'sample_count', 'indices'])
def test_replay_rejects_evidence_drift(damage):
    inner, outer, target, plan, choices = synthetic_evidence()
    row = inner[0]
    if damage == 'train_target':
        row['training_target'][0] += 1
    elif damage == 'validation_target':
        row['target'][0] += 1
    elif damage == 'scale':
        row['normalization_scale'] *= 2
        row['loss'] /= 2
    elif damage == 'prediction':
        row['prediction'][0] += 1
    elif damage == 'seed':
        row['evaluation_seed'] += 1
    elif damage == 'identity':
        row['configuration_id'] = 'another'
    elif damage == 'fit_time':
        row['fit_seconds'] = float('nan')
    elif damage == 'status':
        row['status'] = 'failed'
    elif damage == 'sample_count':
        row['training_samples'] += 1
    else:
        row['training'].reverse()
    with pytest.raises((ValueError, AssertionError)):
        runner.validate_rows(inner, outer, np.empty((65, 0)), target, plan, choices)


def test_summary_rejects_a_choice_changed_after_outer_evaluation():
    inner, outer, target, plan, choices = synthetic_evidence()
    names = tuple(choice.name for choice in choices)
    selections = {}
    for fold in plan:
        rows = [row for row in inner if row['outer_fold'] == fold.ordinal]
        selections[fold.ordinal] = core.choose_inner(rows, names, 3)
    selections[0]['candidate'] = 'C32'
    with pytest.raises(ValueError, match='Locked selection differs'):
        core.summarize(inner, outer, selections, names)

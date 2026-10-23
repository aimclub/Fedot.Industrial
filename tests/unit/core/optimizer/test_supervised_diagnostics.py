"""Follow-up planning is read-only; selection diagnosis opens TRAIN only."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.industrial.experiments.mutation_planning.population_study import supervised_diagnostics as diagnostics
from benchmark.industrial.experiments.mutation_planning.population_study.protocol import (
    experiment_matrix, load_protocol,
)


def _raw():
    return json.loads(Path(diagnostics.__file__).with_name('supervised_diagnostics_defaults.json').read_text())


def test_diagnostic_config_is_frozen_and_preserves_seed_order():
    config = diagnostics.parse_config(_raw())
    assert config.stability_seeds == (17, 29, 101, 211, 307)
    with pytest.raises(FrozenInstanceError):
        config.profile_repeats = 7


@pytest.mark.parametrize('update', [
    {'unexpected': 1}, {'profile_repeats': True}, {'profile_repeats': 6},
    {'stability_seeds': [1, 2]}, {'stability_seeds': [1, 2, 2]},
    {'recheck_ordinals': [True]}, {'profile_datasets': [['nested']]},
])
def test_diagnostic_config_rejects_ambiguous_values(update):
    with pytest.raises(ValueError):
        diagnostics.parse_config({**_raw(), **update})


def test_dry_run_never_creates_output_or_reads_dataset_files(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    protocol = load_protocol()
    (source / 'frozen_manifest.json').write_text(json.dumps({
        'protocol': protocol.to_record(), 'protocol_sha256': protocol.digest, 'matrix': experiment_matrix(protocol),
    }))
    output = tmp_path / 'not-created'
    result = diagnostics.run_diagnostics(source, output)
    assert result['purpose'] == 'diagnostic_only_not_full_acceptance'
    assert len(result['jobs']) == 4
    assert not output.exists()


def test_train_only_loader_does_not_read_malformed_test(tmp_path):
    directory = tmp_path / 'Tiny'
    directory.mkdir()
    (directory / 'Tiny_TRAIN.csv').write_text('1,10,20\n2,30,40\n')
    (directory / 'Tiny_TEST.csv').write_text('this is deliberately not a valid dataset')
    x, y, path = diagnostics._train_arrays(
        {'supervised_root': str(tmp_path)}, {'name': 'Tiny', 'task': 'regression'})
    np.testing.assert_array_equal(x, [[10, 20], [30, 40]])
    np.testing.assert_array_equal(y, [1, 2])
    assert path.name == 'Tiny_TRAIN.csv'


def test_stability_summary_aggregates_validation_only_and_is_order_independent():
    rows = [{'dataset': 'Tiny', 'candidate': 'simple', 'validation_loss': loss} for loss in [1., 2., 3.]]
    summary, = diagnostics.summarize_stability(rows)
    assert summary == {'dataset': 'Tiny', 'candidate': 'simple', 'splits': 3, 'mean_loss': 2., 'sd_loss': 1.}
    assert diagnostics.summarize_stability(list(reversed(rows))) == [summary]


def _selection_record():
    return {
        'status': 'succeeded', 'scenario': {'name': 'Tiny', 'task': 'regression'},
        'seed': 42, 'mode': 'planned', 'ordinal': 0, 'selected_graph': 'winner',
        'objective_calls': [
            {'valid': True, 'graph_id': 'winner', 'fitness': .4},
            {'valid': True, 'graph_id': 'other', 'fitness': .5},
            {'valid': True, 'graph_id': 'winner', 'fitness': .45},
            {'valid': False, 'graph_id': 'failed', 'fitness': None},
        ],
        'test_target': [0., 0., 0., 0., 0., 0.], 'test_prediction': [6., 1., 1., 0., 0., 0.],
        'quality_loss': float(np.sqrt(38 / 6) / 2), 'baseline_loss': 2.,
        'source': {'normalization_scale': 2.},
    }


def test_selection_audit_preserves_repeat_scores_and_replays_tail_errors():
    row, = diagnostics.summarize_selection([_selection_record()])
    assert row['selected_graph_contains_observed_minimum']
    assert row['observed_validation_minimum'] == .4
    assert row['best_other_graph_validation_loss'] == .5
    assert row['selected_graph_validation_losses'] == [.4, .45]
    assert row['repeated_graph_scores'][0]['range'] == pytest.approx(.05)
    assert row['test_loss_replayed'] == pytest.approx(row['test_loss_recorded'])
    assert row['test_top1_squared_error_fraction'] == pytest.approx(36 / 38)
    assert row['test_top5_squared_error_fraction'] == 1.
    assert row['test_largest_error_indices'] == [0, 1, 2, 3, 4]
    assert row['objective_attempts'] == 4


def test_selection_audit_is_not_driven_by_test_values_and_detects_a_wrong_winner():
    raw = _selection_record()
    raw['test_prediction'] = [100.] * 6
    row, = diagnostics.summarize_selection([raw])
    assert row['selected_graph_contains_observed_minimum']
    raw['selected_graph'] = 'other'
    row, = diagnostics.summarize_selection([raw])
    assert not row['selected_graph_contains_observed_minimum']
    assert row['validation_margin_to_other_graph'] == pytest.approx(-.1)


@pytest.mark.parametrize('damage', ['missing_winner', 'bad_score', 'empty', 'length', 'nan', 'scale'])
def test_selection_audit_rejects_incomplete_or_nonfinite_evidence(damage):
    raw = _selection_record()
    if damage == 'missing_winner':
        raw['selected_graph'] = 'not-observed'
    elif damage == 'bad_score':
        raw['objective_calls'][0]['fitness'] = float('nan')
    elif damage == 'empty':
        raw['test_target'] = raw['test_prediction'] = []
    elif damage == 'length':
        raw['test_prediction'].pop()
    elif damage == 'nan':
        raw['test_prediction'][0] = float('nan')
    else:
        raw['source']['normalization_scale'] = 0.
    with pytest.raises(ValueError):
        diagnostics.summarize_selection([raw])


def test_selection_audit_handles_perfect_predictions_and_no_alternative():
    raw = _selection_record()
    raw['objective_calls'] = [raw['objective_calls'][0]]
    raw['test_prediction'] = raw['test_target']
    row, = diagnostics.summarize_selection([raw])
    assert row['best_other_graph_validation_loss'] is None
    assert row['test_top1_squared_error_fraction'] == row['test_top5_squared_error_fraction'] == 0.
    assert row['test_loss_replayed'] == 0.


@pytest.mark.parametrize('prediction', [2e154, 1e308])
def test_selection_audit_rejects_error_sum_or_square_overflow(prediction):
    raw = _selection_record()
    raw['test_prediction'] = [prediction] * 6
    with pytest.raises(ValueError, match='aggregation overflowed'):
        diagnostics.summarize_selection([raw])

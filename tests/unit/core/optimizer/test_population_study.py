from copy import deepcopy
import json

import numpy as np
import pytest
from hypothesis import given, strategies as st

from benchmark.industrial.experiments.mutation_planning.population_study.protocol import (
    experiment_matrix, load_protocol, parse_protocol,
)
from benchmark.industrial.experiments.mutation_planning.population_study.report import assess_study


def _records(protocol):
    return [{**row, 'status': 'succeeded', 'quality_loss': 0.1, 'baseline_loss': 0.5,
             'search_seconds': 5., 'refit_seconds': 1., 'predict_seconds': 0.01, 'peak_rss_bytes': 100000,
             'source': {'test_fixture': True}, 'torch_threads': 1, 'cuda_visible': False,
             'threadpools': [{'num_threads': 1}], 'objective_calls': [{'valid': True, 'fold_count': 1}],
             'diagnostics': {'evaluations_succeeded': 1, 'evaluations_failed': 0},
             'generations': [{'population_size': 4, 'valid_fitness_count': 4, 'new_successes': 1} for _ in range(4)]}
            for row in experiment_matrix(protocol)]


def test_matrix_is_complete_counterbalanced_and_only_mode_changes_within_pair():
    protocol = load_protocol()
    rows = experiment_matrix(protocol)
    assert len(rows) == 54
    assert abs(sum(row['mode'] == 'planned' for row in rows[::2]) - len(rows) / 4) <= .5
    for left, right in zip(rows[::2], rows[1::2]):
        assert {left['mode'], right['mode']} == {'planned', 'legacy'}
        assert {k: v for k, v in left.items() if k not in ('mode', 'ordinal')} == {
            k: v for k, v in right.items() if k not in ('mode', 'ordinal')}


def test_protocol_round_trip_and_caller_ownership():
    protocol = load_protocol()
    raw = protocol.to_record()
    assert parse_protocol(json.loads(json.dumps(raw))) == protocol
    raw['seeds'].append(0)
    assert len(protocol.to_record()['seeds']) == 3


@pytest.mark.parametrize('change', [
    {'seeds': [42]}, {'generations': 2}, {'population_size': False}, {'search_seconds': 0},
    {'scenarios': []}, {'extra': 1}, {'schema_version': 'other'},
])
def test_invalid_study_protocol_is_rejected(change):
    with pytest.raises(ValueError):
        parse_protocol({**load_protocol().to_record(), **change})


def test_okhs_cannot_enter_the_allowlist():
    raw = load_protocol().to_record()
    raw['operations']['ts_forecasting'].append('okhs_fdmd_forecaster')
    with pytest.raises(ValueError, match='excluded'):
        parse_protocol(raw)


@given(st.lists(st.integers(), min_size=54, max_size=54))
def test_healthy_report_is_input_order_independent(keys):
    protocol = load_protocol()
    rows = _records(protocol)
    permuted = [row for _, _, row in sorted(zip(keys, range(54), rows))]
    assert assess_study(protocol, permuted) == assess_study(protocol, rows)
    assert assess_study(protocol, rows)['status'] == 'accepted'


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'failure', 'nan', 'source', 'population',
                                    'reuse', 'cost', 'unobserved_failure'])
def test_bad_or_missing_evidence_cannot_close_the_issue(damage):
    protocol = load_protocol()
    rows = _records(protocol)
    if damage == 'missing':
        rows.pop()
    elif damage == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif damage == 'failure':
        rows[0]['status'] = 'failed'
    elif damage == 'nan':
        rows[0]['quality_loss'] = float('nan')
    elif damage == 'source':
        rows[1]['source'] = {'different': True}
    elif damage == 'population':
        rows[0]['generations'][1]['population_size'] = 1
    elif damage == 'reuse':
        rows[0]['generations'][1]['new_successes'] = 0
    elif damage == 'unobserved_failure':
        rows[0]['diagnostics']['evaluations_failed'] = 1
    else:
        rows[1]['search_seconds'] = 100
    assert assess_study(protocol, rows)['status'] != 'accepted'


def test_quality_and_baseline_guards_are_not_vacuous():
    protocol = load_protocol()
    rows = _records(protocol)
    for row in rows:
        row['quality_loss'] = 0.5
    assert assess_study(protocol, rows)['status'] == 'rejected'
    rows = _records(protocol)
    for row in rows:
        if row['mode'] == 'planned':
            row['quality_loss'] = .9
    assert assess_study(protocol, rows)['status'] == 'rejected'


def test_forecast_split_never_exposes_future_values_to_training(tmp_path):
    from benchmark.industrial.experiments.mutation_planning.population_study.data import load_study_data
    raw = load_protocol().to_record()
    train, test = tmp_path / 'train.csv', tmp_path / 'test.csv'
    train.write_text('id,values\nD1,' + ','.join(map(str, range(30))) + '\n', encoding='utf-8')
    test.write_text('D1,30,31,32,33\n', encoding='utf-8')
    raw.update(forecast_train=str(train), forecast_test=str(test))
    first = load_study_data(raw, {'name': 'D1', 'task': 'ts_forecasting'}, 42)
    test.write_text('D1,900,901,902,903\n', encoding='utf-8')
    second = load_study_data(raw, {'name': 'D1', 'task': 'ts_forecasting'}, 42)
    for field in ('train', 'validation', 'refit'):
        np.testing.assert_equal(getattr(first, field).features.numpy(), getattr(second, field).features.numpy())
        np.testing.assert_equal(getattr(first, field).target.numpy(), getattr(second, field).target.numpy())
    assert first.scale == second.scale
    assert first.source['split'] == {'train_end': 26, 'validation_end': 30, 'test_end': 34}
    np.testing.assert_equal(first.test.target.numpy().reshape(-1), [30, 31, 32, 33])

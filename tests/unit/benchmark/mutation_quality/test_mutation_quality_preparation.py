from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from hypothesis import given, strategies as st
import numpy as np
import pytest

from benchmark.industrial.experiments.mutation_quality.config import load_protocol, parse_protocol
from benchmark.industrial.experiments.mutation_quality.preflight import (
    first_eligible_series, inspect_scenario, preview, supervised_splits, temporal_splits,
)


def payload():
    path = Path(__file__).parents[4] / 'benchmark/industrial/experiments/mutation_quality/protocol.json'
    return json.loads(path.read_text('utf-8'))


@pytest.fixture(scope='module', autouse=True)
def warm_optional_splitter():
    # Keep cold import time outside the generated property deadline.
    __import__('sklearn.model_selection')


def test_matrix_is_balanced_deterministic_and_read_only():
    protocol = load_protocol()
    jobs = protocol.jobs()
    assert len(jobs) == len(set(jobs)) == 225
    assert jobs == protocol.jobs()
    for scenario in protocol.scenarios:
        for seed in protocol.seeds:
            assert {arm for name, arm, value in jobs if name == scenario.name and value == seed} == {
                arm.value for arm in protocol.arms}
    assert preview(protocol)['training_enabled'] is False
    assert preview(protocol)['search_budget_hours'] == 37.5


@pytest.mark.parametrize('change', [
    {'training_enabled': True}, {'workers': True}, {'native_threads': 2}, {'seeds': [42, 42]},
    {'seeds': [True]}, {'arms': ['baseline']}, {'outer_fraction': float('nan')},
    {'candidate_seconds': 1000}, {'unknown': 3},
])
def test_invalid_protocol_is_rejected(change):
    value = payload()
    value.update(change)
    with pytest.raises(ValueError):
        parse_protocol(value)


@pytest.mark.parametrize('path', ['../x_TRAIN.ts', 'C:/x_TRAIN.ts', '/x_TRAIN.ts', 'x_TEST.tsv'])
def test_train_path_boundary(path):
    value = deepcopy(payload())
    value['scenarios'][0]['path'] = path
    with pytest.raises(ValueError):
        parse_protocol(value)


@given(st.integers(min_value=12, max_value=60))
def test_supervised_splits_conserve_rows_and_exclude_outer(per_class):
    protocol = load_protocol()
    y = np.repeat(np.arange(3), per_class)
    result = supervised_splits(y, 'classification', protocol)
    search, outer = set(result['search_indices']), set(result['outer_indices'])
    assert not search & outer
    assert search | outer == set(range(len(y)))
    validation_rows = []
    for fold in result['folds']:
        left, right = set(fold['train_indices']), set(fold['validation_indices'])
        assert not left & right
        assert left | right == search
        assert not (left | right) & outer
        validation_rows.extend(right)
    assert sorted(validation_rows) == sorted(search)
    assert result == supervised_splits(y, 'classification', protocol)


@given(st.integers(min_value=132, max_value=5000))
def test_temporal_splits_do_not_cross_outer_boundary(length):
    result = temporal_splits(length, load_protocol())
    assert result['outer_end'] == length
    for fold in result['folds']:
        assert fold['train_end'] == fold['validation_start'] < fold['validation_end'] <= result['outer_start']
        assert fold['validation_end'] - fold['validation_start'] == result['horizon']


def test_series_selection_is_source_order_not_quality_or_missing_value_repair(tmp_path):
    path = tmp_path / 'Daily-train.csv'
    path.write_text('id,v1,v2,v3,v4\nD1,1,,3,4\nD2,1,2,,\nD3,2,3,4,\nD4,9,8,7,6\n', encoding='utf-8')
    # An internal missing value makes a row ineligible, not a compressed series.
    name, values = first_eligible_series(path, 3)
    assert name == 'D3'
    np.testing.assert_array_equal(values, [2, 3, 4])


def test_missing_source_is_reported_without_fallback(tmp_path):
    scenario = load_protocol().scenarios[0]
    row = inspect_scenario(scenario, tmp_path, load_protocol())
    assert row['status'] == 'missing_input'
    assert row['official_test_used'] is False


def test_read_only_train_input(tmp_path, monkeypatch):
    protocol = load_protocol()
    scenario = replace(protocol.scenarios[0], path='x_TRAIN.tsv')
    x = np.column_stack((np.repeat([0, 1], 20), np.arange(40), np.arange(40) + 1))
    np.savetxt(tmp_path / scenario.path, x, delimiter='\t')
    original = Path.open
    opened = []

    def track(path, *args, **kwargs):
        opened.append(path.name)
        assert 'TEST' not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', track)
    row = inspect_scenario(scenario, tmp_path, protocol)
    assert row['status'] == 'ready_for_split_review'
    assert opened == ['x_TRAIN.tsv', 'x_TRAIN.tsv']


def test_regression_ts_uses_existing_parser_without_flattening_channels(tmp_path):
    from benchmark.industrial.experiments.mutation_quality.preflight import read_source
    scenario = replace(load_protocol().scenarios[6], path='reg_TRAIN.ts')
    (tmp_path / scenario.path).write_text(
        '@problemname reg\n@timestamps false\n@univariate false\n@targetlabel true\n@data\n'
        '1,2,3:4,5,6:1.5\n2,3,4:5,6,7:2.5\n', encoding='utf-8')
    x, y, _ = read_source(scenario, tmp_path, load_protocol())
    assert x.shape == (2, 2, 3)
    np.testing.assert_array_equal(x[0, 1], [4, 5, 6])
    np.testing.assert_array_equal(y, [1.5, 2.5])

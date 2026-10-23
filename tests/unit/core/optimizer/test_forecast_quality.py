from copy import deepcopy
import json

import numpy as np
import pytest
from hypothesis import given, strategies as st

from benchmark.industrial.experiments.mutation_planning.population_study import forecast_quality as runtime
from benchmark.industrial.experiments.mutation_planning.population_study.forecast_quality_protocol import (
    QualityCandidate, load_config, parse_config, select_candidates, temporal_cuts,
)


def test_default_diagnostic_is_bounded_and_preserves_current_ssa_parameters():
    config, candidates = load_config()
    assert config['datasets'] == ['D1', 'D2', 'D3']
    assert config['horizon'] == 14 and config['folds'] == 3
    assert len(candidates) == 13
    assert all('okhs' not in name for candidate in candidates for name in candidate.chain)
    ssa = next(candidate for candidate in candidates if candidate.name == 'ssa_default')
    assert json.loads(ssa.parameters_json) == {'device': 'cpu'}


@pytest.mark.parametrize('damage', [
    {'horizon': True}, {'folds': 0}, {'seeds': [[]]}, {'datasets': [[]]}, {'seeds': [42, 42]},
    {'test_comparators': [[]]}, {'candidates': []}, {'schema_version': 'other'},
    {'candidates': [{'name': 'bad', 'kind': 'industrial', 'chain': ['okhs_fdmd_forecaster']}]},
    {'candidates': [{'name': 'bad', 'kind': 'naive', 'parameters': {'ignored': 1}}]},
])
def test_invalid_quality_protocol_is_rejected(damage):
    config, _ = load_config()
    with pytest.raises(ValueError):
        parse_config({**config, **damage})


@given(st.integers(50, 1000), st.integers(1, 8), st.integers(1, 4))
def test_temporal_folds_only_use_the_prefix_before_their_target(length, horizon, folds):
    cuts = temporal_cuts(length, horizon, folds)
    assert len(cuts) == folds and sorted(cuts) == list(cuts)
    assert cuts[-1] + horizon == length
    assert all(left + horizon == right for left, right in zip(cuts, cuts[1:]))


def _small_study(tmp_path):
    train, test = tmp_path / 'train.csv', tmp_path / 'not-created-test.csv'
    train.write_text('D1,' + ','.join(map(str, range(100))) + '\n', encoding='utf-8')
    config, _ = load_config()
    config.update(train=str(train), test=str(test), datasets=['D1'], seeds=[42], folds=2, horizon=4,
                  candidates=[{'name': 'naive', 'kind': 'naive', 'period': 1},
                              {'name': 'drift', 'kind': 'drift'}], test_comparators=['naive'])
    return config, parse_config(config)


def test_validation_seals_selection_without_opening_test_file(tmp_path, monkeypatch):
    config, candidates = _small_study(tmp_path)
    monkeypatch.setattr(runtime, '_diagnostic_snapshot', lambda: {'revision': 'fixture'})
    audit = {'worktree': 'fixture'}
    monkeypatch.setattr(runtime, '_admit', lambda path: audit)
    monkeypatch.setattr(runtime, 'render_reports', lambda *args: None)
    output = tmp_path / 'report'
    selection = runtime.run_validation(config, candidates, output, audit)
    assert selection['D1']['best'] == 'drift'
    assert selection['D1']['best_industrial'] is None
    assert not (tmp_path / 'not-created-test.csv').exists()
    assert not (output / 'test.json').exists()
    assert (output / 'seal.json').exists()
    # The test stage cannot accept an altered winning candidate.
    result = json.loads((output / 'validation.json').read_text())
    result['selections']['D1']['best'] = 'naive'
    (output / 'validation.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='evidence changed'):
        runtime.run_test(config, candidates, output, audit)


def test_test_stage_uses_frozen_selection_and_cannot_be_repeated(tmp_path, monkeypatch):
    config, candidates = _small_study(tmp_path)
    audit = {'worktree': 'fixture'}
    monkeypatch.setattr(runtime, '_diagnostic_snapshot', lambda: {'revision': 'fixture'})
    monkeypatch.setattr(runtime, '_admit', lambda path: audit)
    monkeypatch.setattr(runtime, 'render_reports', lambda *args: None)
    output = tmp_path / 'report'
    selected = runtime.run_validation(config, candidates, output, audit)
    (tmp_path / 'not-created-test.csv').write_text('D1,100,101,102,103\n', encoding='utf-8')
    assert runtime.run_test(config, candidates, output, audit) == selected
    report = json.loads((output / 'test.json').read_text())
    assert report['full_study_accepted'] is False
    assert {row['candidate'] for row in report['records']} == {'naive', 'drift'}
    with pytest.raises(ValueError, match='already exists'):
        runtime.run_test(config, candidates, output, audit)
    with pytest.raises(ValueError, match='only consume validation'):
        select_candidates(report['records'], config, candidates)


def test_incomplete_or_nonfinite_candidates_are_not_selected():
    config = {'folds': 1, 'seeds': [42], 'datasets': ['D1']}
    candidates = (QualityCandidate('bad', 'naive'), QualityCandidate('good', 'naive'))
    rows = [{'stage': 'validation', 'dataset': 'D1', 'candidate': 'bad', 'fold': 0, 'seed': 42,
             'status': 'succeeded', 'mase': float('nan')},
            {'stage': 'validation', 'dataset': 'D1', 'candidate': 'good', 'fold': 0, 'seed': 42,
             'status': 'succeeded', 'mase': 1.0}]
    before = deepcopy(rows)
    _, selected = select_candidates(rows, config, candidates)
    assert selected['D1']['best'] == 'good'
    assert rows[1] == before[1]
    rows.pop()
    with pytest.raises(ValueError, match='No complete'):
        select_candidates(rows, config, candidates)


def test_reference_solver_matches_numpy_and_does_not_modify_inputs():
    rng = np.random.default_rng(42)
    x = 10000 + rng.normal(size=(80, 4))
    y = (2 * x[:, :1] - x[:, 1:2]) + rng.normal(size=(80, 1))
    before_x, before_y = x.copy(), y.copy()
    head = runtime._ReferenceHead(x, y)
    expected = np.column_stack([np.ones(len(x)), x]) @ np.linalg.lstsq(
        np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    np.testing.assert_allclose(head.predict(x), expected, rtol=1e-10)
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


@pytest.mark.parametrize('seal', [{}, {'manifest.json': '0' * 64},
                                  {'manifest.json': 'x' * 64, 'validation.json': '0' * 64}])
def test_test_stage_rejects_incomplete_or_malformed_seal(tmp_path, seal):
    (tmp_path / 'seal.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='seal must contain exactly'):
        runtime.run_test({}, (), tmp_path, {})


def test_reference_candidate_cannot_be_published_as_best_production_model():
    config = {'folds': 1, 'seeds': [42], 'datasets': ['D1']}
    candidates = (QualityCandidate('probe', 'industrial', reference_solver=True),
                  QualityCandidate('real', 'industrial'))
    rows = [{'stage': 'validation', 'dataset': 'D1', 'candidate': name, 'fold': 0, 'seed': 42,
             'status': 'succeeded', 'mase': loss} for name, loss in [('probe', 0.1), ('real', 0.2)]]
    _, selection = select_candidates(rows, config, candidates)
    assert selection['D1'] == {'best': 'probe', 'best_industrial': 'real'}


def test_mlp_diagnostics_and_actual_training_use_the_requested_seed():
    candidate = QualityCandidate('mlp', 'industrial', ('ssa_forecaster',), json.dumps(
        {'head_policy': 'mlp', 'head_epochs': 1, 'device': 'cpu', 'history_lookback': 0}))
    history = np.sin(np.arange(80) / 4)
    first, diag = runtime.forecast(candidate, history, 4, 11)
    repeated, _ = runtime.forecast(candidate, history, 4, 11)
    changed, other_diag = runtime.forecast(candidate, history, 4, 17)
    assert diag['forecast_head']['random_seed'] == 11
    assert other_diag['forecast_head']['random_seed'] == 17
    np.testing.assert_array_equal(first, repeated)
    assert not np.allclose(first, changed)


def test_linear_diagnostics_describe_the_active_reference_head():
    from fedot_ind.core.models.ts_forecasting.lagged_model.mssa_forecaster import MSSAForecaster
    model = MSSAForecaster(4, head_policy='linear', device='cpu').fit(np.sin(np.arange(80) / 4))
    design, target = model._build_head_training_matrices(model.denoised_series_)
    head = runtime._ReferenceHead(design, target)
    head.weights_[:] = 0
    model.head_ = head
    diag, _ = runtime._linear_diagnostics(model)
    assert diag['coefficient_norm'] == 0 and diag['spectral_radius'] == 0
    expected = np.sqrt(np.mean((head.predict(design.astype(np.float32)) - target.astype(np.float32)) ** 2))
    assert diag['actual_rmse'] == pytest.approx(expected)


def test_local_data_reader_rejects_time_compression_but_allows_trailing_padding(tmp_path):
    path = tmp_path / 'series.csv'
    path.write_text('D1,1,2,3,,\n')
    np.testing.assert_array_equal(runtime.load_histories(path, ['D1'])['D1'], [1, 2, 3])
    path.write_text('D1,1,,3\n')
    with pytest.raises(ValueError, match='Interior missing'):
        runtime.load_histories(path, ['D1'])

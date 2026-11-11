"""TRAIN ownership, native replay boundaries, and complete Ridge-grid summaries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

from hypothesis import given, strategies as st
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from benchmark.industrial.experiments.mutation_planning.population_study.ridge_diagnostics import analysis, runner


def _raw():
    return json.loads(Path(runner.__file__).with_name('defaults.json').read_text())


def test_frozen_grid_matches_approved_choices():
    config = analysis.parse_config(_raw())
    assert config.alphas == (.01, .1, 1, 10, 100, 1000)
    assert config.primary_seeds == (42, 123, 777)
    assert len(config.additional_seeds) == 5
    with pytest.raises(FrozenInstanceError):
        config.reference_alpha = 10


@pytest.mark.parametrize('update', [
    {'alphas': [True, 1]}, {'alphas': [1, 1]}, {'alphas': [float('nan')]},
    {'alphas': [0]}, {'alphas': ['1']}, {'alphas': []}, {'reference_alpha': False},
    {'additional_seeds': [42, 5, 6]}, {'primary_seeds': [True, 2, 3]},
    {'structures': ['simple', 'simple']}, {'scalings': ['none', 'other']},
    {'structures': [[], 'simple']}, {'scalings': ['none', {}]},
    {'validation_fraction': 1}, {'dataset': 'Another'}, {'unexpected': 42},
])
def test_invalid_config_rejected(update):
    with pytest.raises(ValueError):
        analysis.parse_config({**_raw(), **update})


def _matrices():
    training = np.asarray([[1, 1000, 1], [2, 4000, 1], [3, 2000, 1], [4, 3000, 1]], dtype=np.float32)
    target = np.asarray([1., 3., 2., 4.])
    validation = np.asarray([[100, -99999, 1], [1, 2, 1]], dtype=np.float32)
    return training, target, validation


@pytest.mark.parametrize('scaling', ['none', 'standard'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_fit_matches_independent_sklearn_and_preserves_inputs(scaling, dtype):
    x, y, validation = _matrices()
    x, validation = x.astype(dtype), validation.astype(dtype)
    copies = [array.copy() for array in (x, y, validation)]
    result = analysis.fit_head(x, y, validation, scaling=scaling, alpha=10)
    scaler = StandardScaler().fit(x) if scaling == 'standard' else None
    training = x if scaler is None else scaler.transform(x)
    checking = validation if scaler is None else scaler.transform(validation)
    reference = Ridge(alpha=10).fit(training, y)
    np.testing.assert_array_equal(result['validation_prediction'], reference.predict(checking))
    for original, copy in zip((x, y, validation), copies):
        np.testing.assert_array_equal(original, copy)
    if scaler is not None:
        np.testing.assert_array_equal(result['scaler']['mean'], x.mean(axis=0))
        assert result['scaler']['samples_seen'] == len(x)
        assert result['matrix']['dtype'] == str(np.dtype(dtype))
    assert result['matrix']['constant_columns'] == 1
    json.dumps(result, allow_nan=False)


@given(shift=st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False))
def test_validation_shift_cannot_change_fitted_scaler_or_coefficients(shift):
    x, y, validation = _matrices()
    base = analysis.fit_head(x, y, validation, scaling='standard', alpha=1.)
    changed = analysis.fit_head(x, y, validation + shift, scaling='standard', alpha=1.)
    assert changed['scaler'] == base['scaler']
    assert changed['coefficient'] == base['coefficient']
    assert changed['original_intercept'] == base['original_intercept']


@pytest.mark.parametrize('alpha', [.01, 1, 1000])
def test_original_basis_coefficients_reconstruct_prediction(alpha):
    x, y, validation = _matrices()
    result = analysis.fit_head(x, y, validation, scaling='standard', alpha=alpha)
    reconstructed = validation @ result['original_coefficient'] + result['original_intercept']
    np.testing.assert_allclose(reconstructed, result['validation_prediction'], rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize('damage', ['integer', 'nan', 'empty', 'rank3', 'target', 'columns', 'alpha', 'scaling'])
def test_invalid_head_input_rejected(damage):
    x, y, validation = _matrices()
    alpha, scaling = 1, 'none'
    if damage == 'integer':
        x = x.astype(int)
    elif damage == 'nan':
        x[0, 0] = np.nan
    elif damage == 'empty':
        x = x[:0]
    elif damage == 'rank3':
        x = x[:, None, :]
    elif damage == 'target':
        y = y[:-1]
    elif damage == 'columns':
        validation = validation[:, :1]
    elif damage == 'alpha':
        alpha = True
    else:
        scaling = 'validation-fit'
    with pytest.raises(ValueError):
        analysis.fit_head(x, y, validation, alpha=alpha, scaling=scaling)


def _records(config):
    return [{'seed': seed, 'structure': structure, 'scaling': scaling, 'alpha': alpha,
             'validation_loss': 2. if scaling == 'none' else 1., 'train_loss': .1,
             'original_coefficient': [float(seed), 1.]}
            for seed in config.primary_seeds + config.additional_seeds
            for structure in config.structures for scaling in config.scalings for alpha in config.alphas]


def test_summary_complete_grid_paired_deltas_and_no_generalization_claim():
    config = analysis.parse_config(_raw())
    records = _records(config)
    result = analysis.summarize(records, config)
    scaled = [row for row in result['configurations'] if row['scaling'] == 'standard']
    assert len(result['configurations']) == 24
    assert all(row['mean_paired_delta_to_unscaled_alpha1'] == -1. for row in scaled)
    assert all(row['better_than_reference_splits'] == 8 for row in scaled)
    assert not result['test_used'] and not result['full_study_accepted']
    assert 'not_unbiased' in result['interpretation']
    # Floating-point summaries need numerical rather than textual invariance.
    reversed_result = analysis.summarize(list(reversed(records)), config)
    assert result['per_split_selection'] == reversed_result['per_split_selection']
    np.testing.assert_allclose([row['mean_loss'] for row in result['configurations']],
                               [row['mean_loss'] for row in reversed_result['configurations']])


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'nan'])
def test_summary_rejects_incomplete_or_duplicate_evidence(damage):
    config = analysis.parse_config(_raw())
    records = _records(config)
    if damage == 'missing':
        records.pop()
    elif damage == 'duplicate':
        records.append(deepcopy(records[0]))
    else:
        records[0]['validation_loss'] = float('nan')
    with pytest.raises(ValueError):
        analysis.summarize(records, config)


def test_test_guard_blocks_test_even_for_hash_and_allows_train(tmp_path):
    train = tmp_path / 'NaturalGasPricesSentiment_TRAIN.csv'
    test = tmp_path / 'NaturalGasPricesSentiment_TEST.csv'
    train.write_text('1,2')
    test.write_text('must not read')
    with runner.train_only_guard('NaturalGasPricesSentiment') as reads:
        assert train.read_text() == '1,2'
        with pytest.raises(RuntimeError, match='Official TEST access'):
            runner.file_hash(test)
    assert reads == [str(train.resolve())]
    assert test.read_text() == 'must not read'


def test_plan_does_not_read_datasets_or_create_output(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry run must not read datasets')
    monkeypatch.setattr(runner, '_train_arrays', forbidden)
    protocol = Path(__file__).resolve(
    ).parents[4] / 'benchmark/industrial/experiments/mutation_planning/population_study/protocol_20261019.json'
    plan = runner.run(protocol, tmp_path / 'absent')
    assert plan['ridge_fits'] == 192
    assert plan['native_reference_fits'] == 16
    assert plan['standalone_replay_fits'] == 16
    assert plan['total_ridge_fits'] == 224
    assert not plan['official_test_read_or_hashed']
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('structure', ['simple', 'composite'])
def test_native_pipeline_replay_on_synthetic_train_only(structure):
    from benchmark.industrial.experiments.mutation_planning.population_study.protocol import load_protocol
    config = analysis.parse_config(_raw())
    t = np.linspace(0, 6, 32)
    features = np.asarray([np.sin(t + i / 5) + (i + 1) / 8 for i in range(16)])
    target = np.linspace(-.5, .5, 16)
    training, validation = np.arange(12), np.arange(12, 16)
    with runner._runtime():
        x, xv, y, evidence = runner.extract_features(
            load_protocol().to_record(), structure, 42, features, target, training, validation, config)
    assert x.shape[0] == 12 and xv.shape[0] == 4
    np.testing.assert_array_equal(y, target[training])
    assert evidence['native_feature_dtype'] == str(x.dtype)
    assert evidence['replay_max_abs_error'] < config.replay_atol
    assert evidence['native_model_parameters']['alpha'] == config.reference_alpha

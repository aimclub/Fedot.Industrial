from hashlib import sha256
from dataclasses import asdict

import numpy as np
import pytest

from benchmark.industrial.experiments.tsf_extension.planning import (
    TrainRelease, identity, load_profile, load_train_csv, origins, score_window, select_series, setup_search,
)
from benchmark.industrial.protocols.assets import LocalAsset
from benchmark.industrial.protocols.schema import ProtocolError


def test_series_selection_is_deterministic_and_excludes_previous_data():
    collection = load_profile()[0][0]
    x = np.arange(collection.min_history, dtype=float)
    rows = {f'Q{i}': x * i for i in range(1, 8)}
    rows = {**rows, 'short': x[:-1], 'collision': rows['Q5']}
    first = select_series(collection, rows, {'Q2'}, {identity(rows['Q1'])})
    second = select_series(collection, dict(reversed(tuple(rows.items()))), {'Q2'}, {identity(rows['Q1'])})
    assert first == second
    chosen, excluded = first
    assert len(chosen) == 4
    assert set(chosen).isdisjoint({'Q1', 'Q2', 'short', 'collision'})
    assert excluded == {'Q1': 'previous_series', 'Q2': 'previous_series', 'short': 'insufficient_history',
                        'collision': 'duplicate_series'}


def test_search_payload_does_not_include_any_late_assessment_value():
    collection = load_profile()[0][1]
    train = np.arange(200, dtype=float)
    earlier, later = origins(len(train), collection)
    development, used = setup_search(train, collection)
    assert earlier == used and later[0] == len(development)
    assert all(right - left == collection.horizon for left, right in zip((*earlier, *later), (*earlier, *later)[1:]))
    train[later[0]:] = -125
    np.testing.assert_array_equal(development, setup_search(train, collection)[0])


def test_missing_test_role_and_trailing_padding(tmp_path):
    collection = load_profile()[0][0]
    payload = b'V1,V2,V3,V4\nQ1,1,2,\n'
    (tmp_path / collection.filename).write_bytes(payload)
    signature = sha256(payload).hexdigest()
    release = TrainRelease(LocalAsset(collection.filename, signature), 'published train', collection.horizon)
    np.testing.assert_array_equal(load_train_csv(tmp_path, collection, release)['Q1'], [1., 2.])
    invalid = TrainRelease(LocalAsset('Quarterly-test.csv', signature), 'published test', collection.horizon)
    with pytest.raises(ProtocolError, match='separate'):
        load_train_csv(tmp_path, collection, invalid)


def test_zero_scale_does_not_become_a_perfect_mase():
    collection = load_profile()[0][2]
    result = score_window(np.ones(200), np.ones(24), np.full(24, 3.), collection)
    assert result == dict(mase=None, mase_status='zero_scale', rmse=2., smape=100.)


def test_authors_tourism_column_length_is_not_inferred_from_padding(tmp_path):
    collection = load_profile()[0][2]
    payload = b'm1,m2\n2,1\n1979,1980\n1,2\n10,100\n20,\n'
    (tmp_path / collection.filename).write_bytes(payload)
    release = TrainRelease(LocalAsset(collection.filename, sha256(payload).hexdigest()), 'published train',
                           collection.horizon)
    rows = load_train_csv(tmp_path, collection, release)
    np.testing.assert_array_equal(rows['m1'], [10., 20.])
    np.testing.assert_array_equal(rows['m2'], [100.])


def test_runtime_reads_only_one_role_and_refuses_late_values_in_a_development_request(tmp_path, monkeypatch):
    from pathlib import Path
    from benchmark.industrial.experiments.tsf_extension.runtime_contract import Role, load_window_request

    collection = load_profile()[0][0]
    x = np.arange(125, dtype=float)
    earlier, later = origins(len(x), collection)
    dev = tmp_path / 'development'
    dev.mkdir()
    path = dev / 'Q1.npy'
    np.save(path, x[:later[0]], allow_pickle=False)
    case = dict(id='Q1', collection=asdict(collection), seeds=(42, 123), search=earlier, assessment=later,
                development_sha256=sha256(path.read_bytes()).hexdigest())
    seen = []
    real_open = Path.open

    def record(opened, *args, **kwargs):
        seen.append(opened)
        assert 'assessment' not in opened.parts
        return real_open(opened, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', record)
    request = load_window_request(dev, case, seed=42, role=Role.DEVELOPMENT)
    assert request.origins == earlier and seen == [path]
    np.save(path, x, allow_pickle=False)
    case['development_sha256'] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ProtocolError, match='late or missing|cannot support'):
        load_window_request(dev, case, seed=42, role=Role.DEVELOPMENT)


def test_weekly_arima_forecasts_only_from_the_admitted_prefix():
    from types import SimpleNamespace
    from benchmark.industrial.experiments.tsf_extension.legacy_worker import fit_nonseasonal_arima

    history = np.arange(36, dtype=float)
    admitted = []

    class Fit:
        mle_retvals = {'converged': True}

        def forecast(self, horizon):
            return np.zeros(horizon)

    def build(prefix, *, order, trend):
        admitted.append(len(prefix))
        return SimpleNamespace(fit=lambda **_: Fit())

    job = SimpleNamespace(history=history, origins=(15, 18), horizon=13)
    result = fit_nonseasonal_arima(job, {'order': (1, 1, 1), 'trend': 'n', 'maxiter': 50}, build)
    assert admitted == [15, 18]
    assert all(len(row['values']) == 13 for row in result['folds'])


def test_extension_reuses_admission_and_delays_all_assessments(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.tsf_extension import execution

    collections, seeds = load_profile()
    rows = [dict(id=f's{i}', collection=asdict(collection), seeds=seeds)
            for collection in collections for i in range(1, 5)]
    manifest = dict(rows=rows, previous_manifest_sha256='local')
    executor = execution.Executor(tmp_path / 'python', tmp_path / 'runtime', tmp_path / 'receipt')
    monkeypatch.setattr(execution, '_require_tser', lambda *_: None)
    monkeypatch.setattr(execution, 'read_manifest', lambda *_: (manifest, 'matrix'))
    monkeypatch.setattr(execution, '_b_profile', lambda *_: {'calibration_seed': 42})
    monkeypatch.setattr(execution, 'read_checked_asset', lambda *_: None)
    calls = []

    def fake_stage(_executor, bundle, config, case, task, work, stage, **kwargs):
        calls.append((stage, case['collection']['id'], task['case'], task['seed']))
        return dict(stage=stage, task=task, complete=True, exit_code=0)

    monkeypatch.setattr(execution, '_stage', fake_stage)
    args = dict(bundle=tmp_path, previous=tmp_path, executor=executor, tser=tmp_path)
    admission, series = tmp_path / 'admission', tmp_path / 'series'
    assert execution.run('admit', output=admission, **args) == 0
    assert len([item for item in calls if item[0] == 'search']) == 3
    assert execution.run('run', output=series, admission=admission, **args) == 0
    assert len([item for item in calls if item[0] == 'search']) == 24
    assert not any(row[0] == 'assess' for row in calls)
    assert execution.run('assess', output=series, admission=admission, **args) == 0
    assert len([item for item in calls if item[0] == 'assess']) == 12


def test_forecast_stages_reject_unaccepted_regression_before_touching_any_input(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.tsf_extension import execution

    (tmp_path / 'run.json').write_text('{"device":"cuda:0"}')
    (tmp_path / 'assess_status.json').write_text('[]')

    def forbidden(*_args):
        pytest.fail('A failed release gate must be checked before touching the new TRAIN bundle')

    monkeypatch.setattr(execution, 'read_manifest', forbidden)
    with pytest.raises(ProtocolError, match='eight complete TSER'):
        execution.run('admit', bundle=tmp_path, previous=tmp_path, executor=None, tser=tmp_path,
                      output=tmp_path / 'extension')
    assert not (tmp_path / 'extension').exists()

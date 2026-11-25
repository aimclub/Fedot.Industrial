from hashlib import sha256

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

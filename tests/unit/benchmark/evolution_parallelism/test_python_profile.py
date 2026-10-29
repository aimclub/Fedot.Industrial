import csv
import json
import random
import sys

import pytest

from benchmark.industrial.experiments.evolution_parallelism.profiling import PythonProfile, profile_call, profile_rows


def test_profile_preserves_value_randomness_and_restores_owner(tmp_path):
    state = random.getstate()
    expected = [i * i for i in range(10)]
    result = profile_call(lambda: [i * i for i in range(10)], tmp_path / 'run')
    assert result == expected and random.getstate() == state and sys.getprofile() is None
    metadata = json.loads((tmp_path / 'run/profile.json').read_text())
    assert metadata['current_thread_only'] and metadata['diagnostic_run']
    assert metadata['outcome'] == 'succeeded'
    with (tmp_path / 'run/calls.csv').open() as stream:
        assert any(row['function'] == '<listcomp>' for row in csv.DictReader(stream))


def test_exception_is_not_suppressed_and_nested_owner_is_not_replaced(tmp_path):
    with pytest.raises(ValueError, match='original'):
        with PythonProfile(tmp_path / 'failure', label='failure'):
            with pytest.raises(RuntimeError, match='already owns'):
                with PythonProfile(tmp_path / 'nested', label='nested'):
                    pass
            raise ValueError('original')
    assert sys.getprofile() is None and not (tmp_path / 'nested').exists()
    assert json.loads((tmp_path / 'failure/profile.json').read_text())['error_type'] == 'ValueError'
    with pytest.raises(FileExistsError):
        with PythonProfile(tmp_path / 'failure', label='overwrite'):
            pass


def test_rows_preserve_exclusive_and_cumulative_semantics():
    rows = profile_rows({('statistical/a.py', 4, 'feature'): (1, 2, .1, .4, {}),
                         ('other.py', 1, 'outer'): (1, 1, .2, .6, {})})
    assert rows[0]['function'] == 'outer'
    assert rows[1]['category'] == 'statistics'
    assert rows[1]['calls'] == 2 and rows[1]['exclusive_seconds'] == .1

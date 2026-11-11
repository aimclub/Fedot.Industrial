from dataclasses import FrozenInstanceError
import json
import subprocess
import sys

import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.evaluation.contracts import EvaluationPlan, EvaluationTask
from fedot_ind.core.optimizer.evaluation.dask_config import (
    DaskConfigError, DaskEvaluationPlan, DaskExecutionConfig, parse_dask_config,
)


def test_dask_defaults_are_explicit_and_frozen():
    config = parse_dask_config({})
    assert config.to_record() == {
        'retries': 0, 'startup_timeout_seconds': 60, 'restart_timeout_seconds': 30,
        'poll_interval_seconds': .05, 'memory_limit': 'auto', 'lifecycle': 'batch',
    }
    with pytest.raises(FrozenInstanceError):
        config.retries = 1


@given(retries=st.integers(0, 3), seconds=st.floats(min_value=.001, max_value=600),
       memory=st.one_of(st.just('auto'), st.integers(1, 2**50)))
def test_dask_options_survive_json_round_trip(retries, seconds, memory):
    config = DaskExecutionConfig(retries, seconds, seconds, seconds, memory)
    assert parse_dask_config(json.loads(json.dumps(config.to_record(), allow_nan=False))) == config


@pytest.mark.parametrize('name,value', [
    ('retries', -1), ('retries', 4), ('retries', True), ('retries', 1.), ('retries', '1'),
    ('memory_limit', 0), ('memory_limit', -1), ('memory_limit', True),
    ('memory_limit', .3), ('memory_limit', '1GB'), ('memory_limit', 'AUTO'), ('memory_limit', None),
] + [(name, value) for name in ('startup_timeout_seconds', 'restart_timeout_seconds', 'poll_interval_seconds')
     for value in (None, False, 0, -1, '1', float('inf'), float('-inf'), float('nan'))])
def test_direct_and_parsed_options_reject_the_same_invalid_values(name, value):
    for constructor in (lambda: DaskExecutionConfig(**{name: value}), lambda: parse_dask_config({name: value})):
        with pytest.raises(DaskConfigError) as error:
            constructor()
        assert error.value.violations[0].path == name


def test_unknown_and_invalid_options_are_all_reported_in_stable_order():
    with pytest.raises(DaskConfigError) as error:
        parse_dask_config({'unknown': 1, 'retries': 5, 'memory_limit': '1GB', 'poll_interval_seconds': 0})
    assert [(item.code, item.path) for item in error.value.violations] == [
        ('unknown_field', 'unknown'), ('invalid_value', 'retries'),
        ('invalid_value', 'poll_interval_seconds'), ('invalid_value', 'memory_limit'),
    ]


@pytest.mark.parametrize('payload', [None, [], 'auto', 1, False])
def test_dask_options_require_an_object(payload):
    with pytest.raises(DaskConfigError) as error:
        parse_dask_config(payload)
    assert error.value.violations[0].code == 'invalid_section'


def test_dask_plan_is_distinct_and_does_not_overload_local_backend():
    plan = DaskEvaluationPlan((EvaluationTask('a'), EvaluationTask('b')), max_workers=2)
    assert not isinstance(plan, EvaluationPlan)
    assert not hasattr(plan, 'backend')
    assert DaskEvaluationPlan(()).max_workers == 1
    with pytest.raises(FrozenInstanceError):
        plan.max_workers = 3


@pytest.mark.parametrize('options', [
    {'tasks': [EvaluationTask('a')]}, {'tasks': ('a',)},
    {'tasks': (EvaluationTask('same'), EvaluationTask('same'))},
    {'max_workers': 0}, {'max_workers': -1}, {'max_workers': True}, {'max_workers': 1.5},
] + [{name: value} for name in ('candidate_timeout_seconds', 'total_timeout_seconds')
     for value in (0, -1, True, '1', float('inf'), float('nan'))])
def test_dask_plan_validates_tasks_workers_and_deadlines(options):
    with pytest.raises(ValueError):
        DaskEvaluationPlan(**{'tasks': (), **options})


def test_configuration_imports_do_not_require_dask_or_ml_runtime():
    code = '''
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'dask', 'distributed', 'fedot', 'golem', 'torch'}:
            raise AssertionError('runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from fedot_ind.core.optimizer.configuration import normalize_evolution_config
config = normalize_evolution_config({'evaluation_execution': {'backend': 'dask'}})
assert config.evaluation_execution.dask.retries == 0
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr

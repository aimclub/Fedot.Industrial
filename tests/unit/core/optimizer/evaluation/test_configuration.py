import json

import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.configuration import EvolutionConfigError, normalize_evolution_config
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dask_config import DaskExecutionConfig


@given(st.sampled_from(['sequential', 'process', 'dask']),
       st.floats(min_value=0.001, max_value=600, allow_nan=False, allow_infinity=False))
def test_local_deadlines_survive_round_trip(backend, seconds):
    payload = {'evaluation_execution': {'backend': backend, 'candidate_timeout_seconds': seconds,
                                        'total_timeout_seconds': seconds * 2}}
    config = normalize_evolution_config(payload)
    assert config.evaluation_execution.backend is EvaluationBackend(backend)
    assert normalize_evolution_config(json.loads(json.dumps(config.to_record()))) == config


@pytest.mark.parametrize('payload', [
    {'backend': 'threads'}, {'backend': 'legacy', 'candidate_timeout_seconds': 10},
    {'backend': 'process', 'candidate_timeout_seconds': '10'},
    {'backend': 'process', 'candidate_timeout_seconds': False},
    {'backend': 'sequential', 'total_timeout_seconds': -1},
    {'backend': 'sequential', 'total_timeout_seconds': float('inf')},
    {'backend': 'sequential', 'unknown': 1},
])
def test_invalid_execution_settings_are_rejected_at_boundary(payload):
    with pytest.raises(EvolutionConfigError):
        normalize_evolution_config({'evaluation_execution': payload})


@pytest.mark.parametrize('backend', ['legacy', 'sequential', 'process'])
def test_old_canonical_execution_records_remain_unchanged(backend):
    old_record = {'backend': backend, 'candidate_timeout_seconds': None, 'total_timeout_seconds': None}
    config = normalize_evolution_config({'evaluation_execution': old_record})
    assert config.evaluation_execution.dask is None
    assert config.evaluation_execution.to_record() == old_record


@pytest.mark.parametrize('options', [None, {}, {'retries': 3, 'memory_limit': 1024}])
def test_dask_options_default_and_round_trip_through_evolution_boundary(options):
    config = normalize_evolution_config({'evaluation_execution': {'backend': 'dask', 'dask': options}})
    assert config.evaluation_execution.dask == DaskExecutionConfig(**(options or {}))
    assert normalize_evolution_config(json.loads(json.dumps(config.to_record()))) == config


@pytest.mark.parametrize('backend', ['legacy', 'sequential', 'process'])
def test_dask_options_require_explicit_dask_backend(backend):
    with pytest.raises(EvolutionConfigError) as error:
        normalize_evolution_config({'evaluation_execution': {'backend': backend, 'dask': {}}})
    assert [(item.code.value, item.path) for item in error.value.violations] == [
        ('conflicting_fields', 'evaluation_execution.dask')]
    with pytest.raises(ValueError, match='require the dask backend'):
        EvaluationExecutionConfig(EvaluationBackend(backend), dask=DaskExecutionConfig())


def test_direct_execution_config_defaults_dask_but_rejects_untyped_options():
    assert EvaluationExecutionConfig(EvaluationBackend.DASK).dask == DaskExecutionConfig()
    with pytest.raises(ValueError, match='DaskExecutionConfig'):
        EvaluationExecutionConfig(EvaluationBackend.DASK, dask={})


def test_dask_errors_accumulate_with_other_evolution_violations():
    with pytest.raises(EvolutionConfigError) as error:
        normalize_evolution_config({'random_seed': -1, 'evaluation_execution': {
            'backend': 'dask', 'candidate_timeout_seconds': 0,
            'dask': {'retries': 4, 'memory_limit': '1GB', 'extra': True},
        }})
    assert [(item.code.value, item.path) for item in error.value.violations] == [
        ('invalid_type', 'random_seed'),
        ('unknown_field', 'evaluation_execution.dask.extra'),
        ('invalid_value', 'evaluation_execution.dask.retries'),
        ('invalid_value', 'evaluation_execution.dask.memory_limit'),
        ('invalid_value', 'evaluation_execution.candidate_timeout_seconds'),
    ]


@pytest.mark.parametrize('payload', [[], 'auto', False])
def test_malformed_dask_section_has_structured_path(payload):
    with pytest.raises(EvolutionConfigError) as error:
        normalize_evolution_config({'evaluation_execution': {'backend': 'dask', 'dask': payload}})
    assert [(item.code.value, item.path) for item in error.value.violations] == [
        ('invalid_section', 'evaluation_execution.dask')]

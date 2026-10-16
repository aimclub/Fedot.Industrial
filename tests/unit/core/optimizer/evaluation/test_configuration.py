import json

import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.configuration import EvolutionConfigError, normalize_evolution_config
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend


@given(st.sampled_from(['sequential', 'process']),
       st.floats(min_value=0.001, max_value=600, allow_nan=False, allow_infinity=False))
def test_local_deadlines_survive_round_trip(backend, seconds):
    payload = {'evaluation_execution': {'backend': backend, 'candidate_timeout_seconds': seconds,
                                        'total_timeout_seconds': seconds * 2}}
    config = normalize_evolution_config(payload)
    assert config.evaluation_execution.backend is EvaluationBackend(backend)
    assert normalize_evolution_config(json.loads(json.dumps(config.to_record()))) == config


@pytest.mark.parametrize('payload', [
    {'backend': 'threads'}, {'backend': 'dask'}, {'backend': 'legacy', 'candidate_timeout_seconds': 10},
    {'backend': 'process', 'candidate_timeout_seconds': '10'},
    {'backend': 'process', 'candidate_timeout_seconds': False},
    {'backend': 'sequential', 'total_timeout_seconds': -1},
    {'backend': 'sequential', 'total_timeout_seconds': float('inf')},
    {'backend': 'sequential', 'unknown': 1},
])
def test_invalid_execution_settings_are_rejected_at_boundary(payload):
    with pytest.raises(EvolutionConfigError):
        normalize_evolution_config({'evaluation_execution': payload})

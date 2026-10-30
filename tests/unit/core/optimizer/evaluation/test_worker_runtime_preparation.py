import random

import numpy as np
import torch

from fedot_ind.core.optimizer.evaluation.dask_bootstrap import prepare_worker_runtime
from fedot_ind.core.optimizer.evaluation.dispatcher import _GraphEvaluator


def test_preparation_does_not_fit_models_or_change_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Worker preparation invoked a model objective')

    monkeypatch.setattr(_GraphEvaluator, '_evaluate_domain', forbidden)
    state = random.getstate(), torch.random.get_rng_state().clone()
    flags = torch.get_num_threads(), torch.get_num_interop_threads(), torch.are_deterministic_algorithms_enabled()
    numpy_state = np.random.get_state()
    result = prepare_worker_runtime()
    assert result['model_fit_calls'] == 0 and result['torch_flags_preserved']
    assert result['seconds'] >= 0
    assert random.getstate() == state[0] and torch.equal(torch.random.get_rng_state(), state[1])
    assert (torch.get_num_threads(), torch.get_num_interop_threads(),
            torch.are_deterministic_algorithms_enabled()) == flags
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0] and actual[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual[1], numpy_state[1])

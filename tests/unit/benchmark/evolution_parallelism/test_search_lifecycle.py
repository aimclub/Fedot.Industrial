"""Exercise the coordinator with no candidate training or held-out reads."""
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
from benchmark.industrial.experiments.evolution_parallelism import search


@pytest.mark.parametrize('fail', [False, True])
def test_coordinator_closes_monitor_and_never_reads_test_for_incomplete_search(monkeypatch, tmp_path, fail):
    optimizer_module = importlib.import_module('fedot_ind.core.optimizer.IndustrialEvoOptimizer')
    monitoring = importlib.import_module('benchmark.industrial.experiments.evolution_parallelism.monitoring')
    loads, closed = [], []
    native_loader = search.load_data

    def load(*args, **kwargs):
        loads.append(kwargs['include_test'])
        assert not kwargs['include_test']
        return native_loader(*args, **kwargs)

    class Monitor:
        summary = {}

        def __init__(self, root):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

    class Optimizer:
        def __init__(self, objective, initial, *args, **kwargs):
            self.initial = initial
            record = {'fitness': [1.], 'evaluation_context': {'configuration_id': 'one'}}
            self.completed_evaluations = SimpleNamespace(entries=[SimpleNamespace(to_record=lambda: record)])
            self.eval_dispatcher = SimpleNamespace()
            self.evaluated_population = [None] * 15
            self.diagnostics = SimpleNamespace(summary=SimpleNamespace(to_record=lambda: {}))

        def set_iteration_callback(self, callback):
            pass

        def optimise(self, objective):
            if fail:
                raise RuntimeError('candidate infrastructure')
            return self.initial[:1]

    monkeypatch.setattr(search, 'load_data', load)
    monkeypatch.setattr(monitoring, 'ProcessTreeMonitor', Monitor)
    monkeypatch.setattr(optimizer_module, 'IndustrialEvoOptimizer', Optimizer)
    flags = torch.get_num_threads(), torch.get_num_interop_threads(), torch.are_deterministic_algorithms_enabled()
    root = tmp_path / 'run'
    config = SearchConfig('ts_forecasting', 'sequential')
    if fail:
        with pytest.raises(RuntimeError, match='candidate infrastructure'):
            search.run_search(config, root)
        assert not (root / 'summary.json').exists()
    else:
        result = search.run_search(config, root)
        assert result['status'] == 'incomplete' and result['test_loss'] is None
        assert json.loads((root / 'summary.json').read_text())['status'] == 'incomplete'
    assert loads == [False] and closed == [True]
    assert flags == (torch.get_num_threads(), torch.get_num_interop_threads(), torch.are_deterministic_algorithms_enabled())

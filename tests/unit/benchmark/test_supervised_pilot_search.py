from dataclasses import replace
from random import Random

import numpy as np
import pytest

from benchmark.industrial.experiments.supervised_pilot.data import Panel
from benchmark.industrial.experiments.supervised_pilot import execution
from benchmark.industrial.experiments.supervised_pilot.runtime import Job, ValidationFold, evaluate
from benchmark.industrial.experiments.supervised_pilot.space import initial, proposals, rank
from benchmark.industrial.experiments.supervised_pilot.training import Recipe
from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask, ExecutionResult, ExecutionStatus


def test_population_remains_unique_and_order_independent():
    population = initial()
    scored = [(graph, i % 4) for i, graph in enumerate(population)]
    assert rank(scored) == rank(reversed(scored))
    seen = {graph.key for graph in population}
    rng = Random(42)
    for generation in range(1, 9):
        new = proposals(population, seen, 8, tune=generation in {4, 8}, rng=rng)
        keys = {graph.key for graph in new}
        assert len(keys) == 8 and not keys & seen
        if generation in {4, 8}:
            assert all(any(branch.parameter for branch in graph.branches) for graph in new)
        seen.update(keys)
        population += new


@pytest.mark.parametrize('graph', initial(), ids=lambda graph: '/'.join(graph.branches[0].operations))
def test_every_initial_graph_scores_in_fedot(graph, monkeypatch):
    import torch
    from benchmark.industrial.experiments.supervised_pilot import training

    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    profile = replace(training.training_policy(), min_epochs=1, max_epochs=1, width=4, depth=1)
    monkeypatch.setattr(training, 'training_policy', lambda: profile)
    rng = np.random.default_rng(123)
    x = rng.normal(size=(20, 1, 96))
    y = np.array(['a', 'b'] * 10)
    panel = Panel('Synthetic', np.arange(len(y)), x, y)
    fold = ValidationFold(panel.take(tuple(range(16))), panel.take((16, 17, 18, 19)), (12, 13, 14, 15))
    recipe = {branch.head: Recipe(0.001, 8) for branch in graph.branches}
    try:
        result = evaluate(EvaluationTask(graph.key, 42), Job(graph, (fold,), ('a', 'b'), recipe, 'cpu'))
    finally:
        torch.set_num_threads(prior)
    assert 0 <= result['loss'] <= 1
    assert result['folds'][0]['keys'] == [16, 17, 18, 19]


def test_search_writes_native_history_and_never_submits_assessment(tmp_path, monkeypatch):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from benchmark.industrial.experiments.supervised_pilot.protocol import tsc_protocol

    panel = Panel('Synthetic', np.arange(6), np.ones((6, 1, 8)), np.array(['a', 'b'] * 3))
    monkeypatch.setattr(execution, 'read_development', lambda *_: (
        panel, dict(classes=('a', 'b'), inner=[dict(fit=[0, 1, 2, 3], score=[4, 5], epoch_validation=[2, 3])]),
    ))

    def fake_batch(_session, jobs, budget, seed, deadline, evaluator, attempts):
        assert len(jobs) in (8, 10)
        for _, job in jobs:
            assert len(job.folds) == 1
            assert tuple(job.folds[0].score.rows) == (4, 5)
        return tuple(ExecutionResult(key, ExecutionStatus.SUCCEEDED, 0, {'loss': int(key[:6], 16) / 16**6})
                     for key, _ in jobs)

    monkeypatch.setattr(execution, '_batch', fake_batch)
    result = execution.search(None, 'unused', tmp_path, tsc_protocol().search, 42, 'development',
                              dict(development_sha256='development', recipes={}))
    assert result['status'] == 'complete'
    assert len(result['generations']) == 9
    assert sum(row['evaluated'] for row in result['generations']) == 74
    assert len(OptHistory.load(str(tmp_path / 'opt_history.json')).generations) == 9


def test_worker_receives_the_job_and_seed_in_one_payload(monkeypatch):
    from time import time
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.evaluation import dask_executor
    from fedot_ind.core.optimizer.evaluation.executors import _invoke
    from benchmark.industrial.experiments.supervised_pilot.protocol import tsc_protocol

    def call_as_worker(plan, payloads, evaluator, **kwargs):
        return [_invoke(task, payload, evaluator) for task, payload in zip(plan.tasks, payloads, strict=True)]

    monkeypatch.setattr(dask_executor, 'execute_dask_plan', call_as_worker)
    result, = execution._batch(SimpleNamespace(max_workers=1), (('graph', 9),), tsc_protocol().search, 42,
                               time() + 10, lambda task, payload: task.seed + payload, [])
    assert result.status is ExecutionStatus.SUCCEEDED and result.value == 51

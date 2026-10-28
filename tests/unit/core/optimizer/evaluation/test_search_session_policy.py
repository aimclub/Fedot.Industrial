from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dask_config import DaskExecutionConfig, DaskLifecycle, DaskEvaluationPlan, parse_dask_config
from fedot_ind.core.optimizer.evaluation.dask_dispatcher import DaskEvaluationDispatcher, dask_session_scope
from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask
from fedot_ind.core.optimizer.evaluation.executors import CancellationToken


@pytest.mark.parametrize('value', list(DaskLifecycle))
def test_lifecycle_round_trip(value):
    config = DaskExecutionConfig(lifecycle=value)
    assert parse_dask_config(config.to_record()) == config


def test_json_lifecycle_reaches_evolution_config():
    from fedot_ind.core.optimizer.configuration import normalize_evolution_config
    config = normalize_evolution_config({'evaluation_execution': {'backend': 'dask', 'dask': {'lifecycle': 'search'}}})
    assert config.evaluation_execution.dask.lifecycle is DaskLifecycle.SEARCH


def test_explicit_caller_session_is_not_adopted_or_closed():
    from fedot_ind.core.optimizer.evaluation.dask_executor import DaskEvaluationSession
    config = DaskExecutionConfig(lifecycle=DaskLifecycle.SEARCH)
    session = DaskEvaluationSession(1, config)
    session.client = SimpleNamespace(status='running')
    dispatcher = DaskEvaluationDispatcher.__new__(DaskEvaluationDispatcher)
    dispatcher.config = EvaluationExecutionConfig(EvaluationBackend.DASK, dask=config)
    with dask_session_scope(session):
        with dispatcher.search_scope():
            assert dispatcher._search_session(DaskEvaluationPlan((EvaluationTask('one'),))) is session
            assert dispatcher._search_resources.session is None
    assert session.client.status == 'running'


def test_public_optimizer_owns_scope_even_when_search_raises():
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.domain import EvolutionPhase
    events = []

    @contextmanager
    def scope():
        events.append('enter')
        try:
            yield
        finally:
            events.append('close')

    optimizer = IndustrialEvoOptimizer.__new__(IndustrialEvoOptimizer)
    optimizer.eval_dispatcher = SimpleNamespace(search_scope=scope)
    optimizer.evolution_state = SimpleNamespace(phase=EvolutionPhase.COMPLETED)

    def fail(objective):
        events.append('search')
        raise RuntimeError('search failure')

    optimizer._optimise = fail
    with pytest.raises(RuntimeError, match='search failure'):
        optimizer.optimise(lambda graph: None)
    assert events == ['enter', 'search', 'close']


def test_lazy_startup_cannot_receive_an_extra_batch_budget(monkeypatch, tmp_path):
    import fedot_ind.core.optimizer.evaluation.dask_dispatcher as module
    clock, closed = [0.], []

    class Session:
        def __init__(self, *args, **kwargs):
            assert kwargs['startup_timeout'] == 4

        def __enter__(self):
            clock[0] += 5
            return self

        def __exit__(self, *args):
            closed.append(True)

    monkeypatch.setattr(module, 'DaskEvaluationSession', Session)
    monkeypatch.setattr(module, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module, 'execute_dask_plan', lambda *a, **k: pytest.fail('expired budget executed'))
    dispatcher = DaskEvaluationDispatcher.__new__(DaskEvaluationDispatcher)
    dispatcher.config = EvaluationExecutionConfig(EvaluationBackend.DASK,
                                                  dask=DaskExecutionConfig(lifecycle=DaskLifecycle.SEARCH))
    dispatcher.cancellation = CancellationToken()
    with dispatcher.search_scope():
        result, = dispatcher._execute_plan(DaskEvaluationPlan((EvaluationTask('one'),), total_timeout_seconds=4), [1])
    assert result.status.value == 'timed_out' and closed == [True]


@pytest.mark.parametrize('fail', [False, True])
def test_search_creates_one_lazy_session_and_closes_on_every_exit(monkeypatch, fail):
    import fedot_ind.core.optimizer.evaluation.dask_dispatcher as module
    events = []

    class Session:
        def __init__(self, *a, **k):
            events.append('create')

        def __enter__(self):
            events.append('enter')
            return self

        def __exit__(self, *a):
            events.append('close')

    monkeypatch.setattr(module, 'DaskEvaluationSession', Session)
    dispatcher = DaskEvaluationDispatcher.__new__(DaskEvaluationDispatcher)
    dispatcher.config = EvaluationExecutionConfig(EvaluationBackend.DASK,
                                                  dask=DaskExecutionConfig(lifecycle=DaskLifecycle.SEARCH))
    dispatcher.cancellation = CancellationToken()
    plan = DaskEvaluationPlan((EvaluationTask('one'),))

    def run():
        with dispatcher.search_scope():
            assert not events
            assert dispatcher._search_session(plan) is dispatcher._search_session(plan)
            if fail:
                raise RuntimeError('body')

    if fail:
        with pytest.raises(RuntimeError, match='body'):
            run()
    else:
        run()
    assert events == ['create', 'enter', 'close']
    assert dispatcher._search_resources is None

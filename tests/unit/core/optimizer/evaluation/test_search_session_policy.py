from types import SimpleNamespace
from dataclasses import replace

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

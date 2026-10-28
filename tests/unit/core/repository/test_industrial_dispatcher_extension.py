from types import SimpleNamespace

import pytest
from fedot.extensions import clear_extension_registry, get_registered_extensions
from golem.core.optimisers.fitness import null_fitness

from fedot_ind.core.optimizer.domain import EVALUATION_OUTCOME_SCHEMA
from fedot_ind.core.repository.IndustrialDispatcher import IndustrialDispatcher
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder


def test_direct_legacy_evaluation_activates_extension_inside_candidate_scope():
    clear_extension_registry()
    observed_registry_sizes = []
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher._adapter = SimpleNamespace(adapt_func=lambda function: function)
    dispatcher.logger = SimpleNamespace(info=lambda message: None)

    def evaluate(graph):
        observed_registry_sizes.append(len(get_registered_extensions()))
        return null_fitness(), graph

    dispatcher._evaluate_graph = evaluate
    result = dispatcher.eval_ind("graph", "individual")

    assert result.uid_of_individual == "individual"
    assert observed_registry_sizes == [1]
    assert get_registered_extensions() == ()


def test_dispatcher_records_success_failure_and_reused_evaluations():
    recorder = EvolutionDiagnosticsRecorder()
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher.diagnostics_recorder = recorder
    successful = SimpleNamespace(
        uid_of_individual="successful",
        graph=SimpleNamespace(descriptive_id="g1"),
        metadata={"computation_time_in_seconds": 0.25},
    )
    failed = SimpleNamespace(
        uid_of_individual="failed",
        graph=SimpleNamespace(descriptive_id="g2"),
        metadata={
            "computation_time_in_seconds": 0.75,
            "evaluation_error": "ValueError('bad graph')",
            "evaluation_error_type": "ValueError",
        },
    )
    reused = SimpleNamespace(
        uid="reused",
        graph=SimpleNamespace(descriptive_id="g3"),
    )

    dispatcher._record_evaluation_results((successful, failed, None))
    dispatcher._record_reused_individuals((reused,))
    summary = recorder.snapshot().summary

    assert summary.evaluations_succeeded == 1
    assert summary.evaluations_failed == 1
    assert summary.evaluations_reused == 1
    assert summary.evaluation_failure_counts == {"ValueError": 1}
    assert summary.evaluation_duration_seconds == 1.0


def test_dispatcher_worker_state_excludes_coordinator_recorder():
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher.diagnostics_recorder = EvolutionDiagnosticsRecorder()
    dispatcher.completed_evaluation_callback = lambda results: None
    dispatcher.marker = "kept"

    state = dispatcher.__getstate__()

    assert state["diagnostics_recorder"] is None
    assert state["completed_evaluation_callback"] is None
    assert state["marker"] == "kept"
    assert dispatcher.diagnostics_recorder is not None
    assert dispatcher.completed_evaluation_callback is not None


def test_null_fitness_is_recorded_as_structured_evaluation_failure():
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher._adapter = SimpleNamespace(adapt_func=lambda function: function)
    dispatcher.logger = SimpleNamespace(info=lambda message: None)
    dispatcher._evaluate_graph = lambda graph: (null_fitness(), graph)

    result = dispatcher.eval_ind("graph", "individual")

    assert result.metadata["evaluation_error_type"] == "InvalidFitness"
    assert "invalid fitness" in result.metadata["evaluation_error"]
    assert result.metadata["evaluation_schema"] == EVALUATION_OUTCOME_SCHEMA
    assert result.metadata["evaluation_status"] == "failed"
    assert result.metadata["evaluation_failure"]["code"] == "invalid_fitness"


def test_dispatcher_preserves_native_fedot_failure_instead_of_invalid_fitness():
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
    from fedot.core.optimisers.objective.evaluation_contracts import (
        EvaluationFailure, EvaluationIncomplete, FailureKind,
    )

    evaluator = object.__new__(PipelineObjectiveEvaluateWithTensorData)
    evaluator.last_outcome = EvaluationIncomplete(
        'graph', 0, (), (), EvaluationFailure(FailureKind.DATA, 'no validation data'))
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher._adapter = SimpleNamespace(adapt_func=lambda function: function)
    dispatcher.logger = SimpleNamespace(info=lambda message: None)
    dispatcher._objective_eval = evaluator.evaluate
    dispatcher._evaluate_graph = lambda graph: (null_fitness(), graph)

    result = dispatcher.eval_ind('graph', 'individual')

    assert result.metadata['evaluation_failure'] == {
        'code': 'objective_incomplete',
        'error_type': 'FedotEvaluation:data',
        'message': 'data: no validation data',
    }


@pytest.mark.parametrize("error_type", [ValueError, AssertionError])
def test_expected_objective_error_is_preserved_as_typed_failure(error_type):
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher._adapter = SimpleNamespace(adapt_func=lambda function: function)
    dispatcher.logger = SimpleNamespace(info=lambda message: None)

    def fail_evaluation(graph):
        raise error_type(f"cannot evaluate {graph}")

    dispatcher._evaluate_graph = fail_evaluation

    result = dispatcher.eval_ind("graph", "individual")

    assert not result.fitness.valid
    assert result.metadata["evaluation_failure"] == {
        "code": "objective_exception",
        "error_type": error_type.__name__,
        "message": repr(error_type("cannot evaluate graph")),
    }


@pytest.mark.parametrize("error_type", [OSError, ImportError, MemoryError])
def test_unexpected_environment_error_is_not_silently_converted_to_null_fitness(error_type):
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher._adapter = SimpleNamespace(adapt_func=lambda function: function)
    dispatcher.logger = SimpleNamespace(info=lambda message: None)

    def fail_evaluation(graph):
        raise error_type(f"storage unavailable for {graph}")

    dispatcher._evaluate_graph = fail_evaluation

    with pytest.raises(error_type, match="storage unavailable"):
        dispatcher.eval_ind("graph", "individual")


def test_legacy_population_is_sequential_without_dask_and_records_reuse(monkeypatch):
    import dask
    from golem.core.adapter import DirectAdapter
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.graph import OptGraph, OptNode
    from golem.core.optimisers.opt_history_objects.individual import Individual

    def unexpected_scheduler(*args, **kwargs):
        pytest.fail('legacy evaluation must not use a Dask scheduler')

    monkeypatch.setattr(dask, 'compute', unexpected_scheduler)
    monkeypatch.setattr(dask, 'delayed', unexpected_scheduler)
    observed, completions = [], []
    recorder = EvolutionDiagnosticsRecorder()
    dispatcher = IndustrialDispatcher(DirectAdapter(), n_jobs=2, diagnostics_recorder=recorder,
                                      completed_evaluation_callback=lambda results: completions.extend(results))
    a, b = [Individual(OptGraph(OptNode(name))) for name in ('ridge', 'dtreg')]
    reused = Individual(OptGraph(OptNode('rfreg')), fitness=SingleObjFitness(.5))

    def evaluate(graph):
        observed.append(graph.descriptive_id)
        return SingleObjFitness(1.)

    dispatcher.dispatch(evaluate)
    result = dispatcher.evaluate_population([a, b, reused])
    assert result == [a, b, reused]
    assert observed == [a.graph.descriptive_id, b.graph.descriptive_id]
    assert [item.uid_of_individual for item in completions] == [a.uid, b.uid]
    assert all(not hasattr(item, 'compute') for item in completions)
    assert recorder.snapshot().summary.evaluations_succeeded == 2
    assert recorder.snapshot().summary.evaluations_reused == 1


def test_legacy_untimed_retry_uses_direct_results_and_stops_after_success(monkeypatch):
    import dask

    monkeypatch.setattr(dask, 'compute', lambda *args: pytest.fail('unexpected Dask compute'))
    dispatcher = object.__new__(IndustrialDispatcher)
    individuals = [SimpleNamespace(graph=name, uid=name) for name in ('bad', 'good', 'unused')]
    calls, recorded = [], []

    def evaluate(owner, **kwargs):
        assert owner is dispatcher
        calls.append(kwargs)
        return kwargs['uid_of_individual']

    dispatcher.industrial_evaluate_single = evaluate
    dispatcher._record_evaluation_results = lambda results: recorded.extend(results)
    dispatcher.apply_evaluation_results = lambda population, results: population if results == ['good'] else []

    assert dispatcher._eval_at_least_one(individuals) == [individuals[1]]
    assert recorded == ['bad', 'good']
    assert [call['uid_of_individual'] for call in calls] == ['bad', 'good']
    assert all(call['with_time_limit'] is False for call in calls)

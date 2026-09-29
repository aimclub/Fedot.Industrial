from types import SimpleNamespace

from fedot.extensions import clear_extension_registry, get_registered_extensions
from golem.core.optimisers.fitness import null_fitness

from fedot_ind.core.repository.IndustrialDispatcher import IndustrialDispatcher
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder


def test_dask_evaluation_activates_extension_inside_worker_scope():
    clear_extension_registry()
    observed_registry_sizes = []
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher._adapter = SimpleNamespace(adapt_func=lambda function: function)
    dispatcher.logger = SimpleNamespace(info=lambda message: None)

    def evaluate(graph):
        observed_registry_sizes.append(len(get_registered_extensions()))
        return null_fitness(), graph

    dispatcher._evaluate_graph = evaluate
    result = dispatcher.eval_ind(
        "graph", "individual").compute(scheduler="synchronous")

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
    dispatcher.marker = "kept"

    state = dispatcher.__getstate__()

    assert state["diagnostics_recorder"] is None
    assert state["marker"] == "kept"
    assert dispatcher.diagnostics_recorder is not None


def test_null_fitness_is_recorded_as_structured_evaluation_failure():
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher._adapter = SimpleNamespace(adapt_func=lambda function: function)
    dispatcher.logger = SimpleNamespace(info=lambda message: None)
    dispatcher._evaluate_graph = lambda graph: (null_fitness(), graph)

    result = dispatcher.eval_ind(
        "graph", "individual").compute(scheduler="synchronous")

    assert result.metadata["evaluation_error_type"] == "InvalidFitness"
    assert "invalid fitness" in result.metadata["evaluation_error"]

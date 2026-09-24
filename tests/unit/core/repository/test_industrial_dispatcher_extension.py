from types import SimpleNamespace

from fedot.extensions import clear_extension_registry, get_registered_extensions
from golem.core.optimisers.fitness import null_fitness

from fedot_ind.core.repository.IndustrialDispatcher import IndustrialDispatcher


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
    result = dispatcher.eval_ind("graph", "individual").compute(scheduler="synchronous")

    assert result.uid_of_individual == "individual"
    assert observed_registry_sizes == [1]
    assert get_registered_extensions() == ()

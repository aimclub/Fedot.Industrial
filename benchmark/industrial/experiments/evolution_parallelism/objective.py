"""Do not let GOLEM's legacy exception fallback erase the native fold failure."""
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.objective.objective import to_fitness


class SearchObjective(Objective):
    def evaluate_strict(self, graph, **kwargs):
        return to_fitness(tuple(metric(graph, **kwargs) for _, metric in self.metrics), self.is_multi_objective)

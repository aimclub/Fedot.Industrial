"""One bounded parameter change; no fitting, hidden tuning or global configuration."""
from copy import deepcopy
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class BoundedParameterMutation:
    grids: tuple

    @property
    def __name__(self):
        return 'parameter_change_mutation'

    def _choices(self, index, name, current):
        from fedot_ind.core.tuning.generator_catalog import parameter_is_active
        for grid in self.grids:
            if grid.operation == name and parameter_is_active(name, grid.parameter, current):
                for value in grid.values:
                    previous = current.get(grid.parameter)
                    if grid.parameter not in current or (type(previous), previous) != (type(value), value):
                        yield index, grid.parameter, value

    def has_candidates(self, graph):
        return any(next(self._choices(index, node.content['name'], node.content.get('params') or {}), None)
                   is not None for index, node in enumerate(graph.nodes))

    def __call__(self, pipeline, **kwargs):
        candidates = []
        for index, node in enumerate(pipeline.nodes):
            name = node.operation.operation_type
            current = node.parameters if isinstance(node.parameters, dict) else {}
            candidates.extend(self._choices(index, name, current))
        if not candidates:
            return pipeline
        index, parameter, value = random.choice(candidates)
        result = deepcopy(pipeline)
        node = result.nodes[index]
        current = node.parameters if isinstance(node.parameters, dict) else {}
        node.parameters = {**current, parameter: value}
        from fedot_ind.core.tuning.generator_catalog import generator_space
        space = generator_space(node.operation.operation_type)
        if space is not None:
            node.parameters = space.normalize(node.parameters)
        return result


def rewrite_subgraph_mutation(*args, **kwargs):
    raise RuntimeError('Subgraph mutation requires the capability-aware PlannedMutation interpreter')

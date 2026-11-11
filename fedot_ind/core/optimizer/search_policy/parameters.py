"""One bounded parameter change; no fitting, hidden tuning or global configuration."""
from copy import deepcopy
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class BoundedParameterMutation:
    grids: tuple
    eligible: object = None

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
        if self.eligible is not None:
            for index, node in enumerate(graph.nodes):
                for _, parameter, value in self._choices(index, node.content['name'], node.content.get('params') or {}):
                    proposal = deepcopy(graph)
                    changed = proposal.nodes[index]
                    changed.content['params'] = self._normalize(changed.content['name'],
                                                                {**(changed.content.get('params') or {}), parameter: value})
                    if self.eligible(proposal):
                        return True
            return False
        return any(next(self._choices(index, node.content['name'], node.content.get('params') or {}), None)
                   is not None for index, node in enumerate(graph.nodes))

    @staticmethod
    def _normalize(name, values):
        from fedot_ind.core.tuning.generator_catalog import generator_space
        space = generator_space(name)
        return space.normalize(values) if space is not None else values

    def _proposal(self, pipeline, choice):
        index, parameter, value = choice
        result = deepcopy(pipeline)
        node = result.nodes[index]
        current = node.parameters if isinstance(node.parameters, dict) else {}
        node.parameters = self._normalize(node.operation.operation_type, {**current, parameter: value})
        return result

    def __call__(self, pipeline, **kwargs):
        candidates = []
        for index, node in enumerate(pipeline.nodes):
            name = node.operation.operation_type
            current = node.parameters if isinstance(node.parameters, dict) else {}
            candidates.extend(self._choices(index, name, current))
        if not candidates:
            return pipeline
        if self.eligible is None:
            return self._proposal(pipeline, random.choice(candidates))
        random.shuffle(candidates)
        for choice in candidates:
            proposal = self._proposal(pipeline, choice)
            if self.eligible(proposal):
                return proposal
        return pipeline


def rewrite_subgraph_mutation(*args, **kwargs):
    raise RuntimeError('Subgraph mutation requires the capability-aware PlannedMutation interpreter')

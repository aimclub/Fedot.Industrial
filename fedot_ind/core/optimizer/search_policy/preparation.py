"""Parameter-aware preparation for the new protocol, not a legacy override change."""
from copy import deepcopy
from dataclasses import dataclass
import json

from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json, pipeline_configuration
from .contracts import ParameterGrid


@dataclass(frozen=True)
class ParameterSearchPreparation:
    adapter: object
    defaults_json: str
    grids: tuple
    task: str
    normalize_generators: bool = False

    def __post_init__(self):
        if type(self.normalize_generators) is not bool:
            raise ValueError('normalize_generators must be a boolean')
        if self.task not in ('classification', 'regression', 'ts_forecasting'):
            raise ValueError('Unsupported search task')
        defaults = json.loads(self.defaults_json)
        if not isinstance(defaults, dict) or canonical_json(defaults) != self.defaults_json:
            raise ValueError('Defaults must be a canonical JSON object')
        if not isinstance(self.grids, tuple) or any(not isinstance(grid, ParameterGrid) for grid in self.grids):
            raise ValueError('Grids must be typed ParameterGrid values')
        for grid in self.grids:
            if grid.parameter not in defaults.get(grid.operation, {}):
                raise ValueError('Every searched parameter needs an explicit canonical default')
            value = defaults[grid.operation][grid.parameter]
            if (type(value), value) not in {(type(item), item) for item in grid.values}:
                raise ValueError('A canonical default must belong to its parameter grid')

    @classmethod
    def from_defaults(cls, adapter, defaults, grids, task, *, normalize_generators=False):
        if normalize_generators:
            from fedot_ind.core.tuning.generator_catalog import generator_space
            defaults = {name: {**(generator_space(name).defaults if generator_space(name) else {}), **values}
                        for name, values in defaults.items()}
        return cls(adapter, canonical_json(defaults), grids, task, normalize_generators)

    def __call__(self, graph):
        from fedot_ind.integration.fedot.extensions import industrial_extension_scope
        from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
        from fedot_ind.integration.fedot.extensions.contracts import IndustrialRuntimeInterface

        defaults = json.loads(self.defaults_json)
        declarations = {operation.name: operation for operation in load_industrial_extension_catalog().operations}
        with industrial_extension_scope():
            pipeline = self.adapter.restore(deepcopy(graph))
            for node in pipeline.nodes:
                name = node.operation.operation_type
                current = node.parameters if isinstance(node.parameters, dict) else {}
                parameters = {**current, **defaults.get(name, {})}
                for grid in self.grids:
                    if grid.operation == name and grid.parameter in current:
                        value = current[grid.parameter]
                        if (type(value), value) not in {(type(item), item) for item in grid.values}:
                            raise ValueError(f'Parameter {name}.{grid.parameter} is outside its declared grid')
                        parameters[grid.parameter] = value
                declaration = declarations.get(name)
                if self.normalize_generators:
                    from fedot_ind.core.tuning.generator_catalog import generator_space
                    space = generator_space(name)
                    if space is not None:
                        parameters = space.normalize(parameters)
                if (declaration is not None and declaration.runtime_interface is IndustrialRuntimeInterface.INPUT_DATA
                        and len(declaration.tasks) > 1 and self.task in declaration.tasks):
                    if 'task_type' in parameters and parameters['task_type'] != self.task:
                        raise ValueError('Explicit task_type conflicts with the graph task')
                    parameters['task_type'] = self.task
                node.parameters = json.loads(canonical_json(parameters))
            configuration = canonical_json(pipeline_configuration(pipeline, self.task))
            return PreparedEvaluationGraph(self.adapter.adapt(pipeline), configuration)

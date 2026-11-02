"""Preserve FEDOT's typed failure at the Industrial worker boundary."""

from functools import partial
from dataclasses import dataclass

from fedot.core.data.tensor_data import TensorData
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
from fedot.core.optimisers.objective.evaluation_contracts import EvaluationIncomplete

from fedot_ind.core.optimizer.domain import EvaluationFailure, EvaluationFailureCode
from fedot_ind.core.optimizer.mutation_planning import RawInputSpec


@dataclass(frozen=True)
class ConfirmedObjective:
    """Explicit homogeneous training context, including through measured wrappers."""

    objective: object
    raw_input: RawInputSpec

    def __post_init__(self):
        if not callable(self.objective) or not isinstance(self.raw_input, RawInputSpec):
            raise TypeError('A confirmed objective requires a callable and typed raw input')

    @classmethod
    def from_training_data(cls, objective, training_data):
        if not isinstance(training_data, tuple) or not training_data:
            raise ValueError('Pass an explicit non-empty tuple of training inputs, not a fold producer')
        contexts, tasks = set(), set()
        for data in training_data:
            if not isinstance(data, TensorData):
                raise TypeError('Only homogeneous TensorData inputs can confirm this context')
            data_type = getattr(data.data_type, 'name', data.data_type)
            normalized = {'table': 'tabular', 'tabular': 'tabular', 'ts': 'ts', 'multi_ts': 'ts', 'image': 'image'}
            if data_type not in normalized:
                raise ValueError('Unsupported or multimodal training input')
            contexts.add(RawInputSpec(normalized[data_type], data.target is not None))
            tasks.add(data.task.task_type)
        if len(contexts) != 1 or len(tasks) != 1:
            raise ValueError('Training folds must share one input and task contract')
        return cls(objective, next(iter(contexts)))

    def __call__(self, graph):
        return self.objective(graph)


def industrial_failure_from_fedot(outcome) -> EvaluationFailure | None:
    """Translate an incomplete evaluation; do not reinterpret other outcomes."""
    if not isinstance(outcome, EvaluationIncomplete):
        return None
    failure = outcome.failure
    return EvaluationFailure(
        code=EvaluationFailureCode.OBJECTIVE_INCOMPLETE,
        error_type=failure.exception_type or f"FedotEvaluation:{failure.kind.value}",
        message=f"{failure.kind.value}: {failure.message}",
    )


def confirmed_input_from_objective(objective) -> RawInputSpec | None:
    """Read a native homogeneous source, never consume a custom fold iterator.

    FEDOT's TensorData evaluator exposes the first argument of its partial
    producer as tensor_data. Unknown producers retain conservative planning.
    """
    if isinstance(objective, ConfirmedObjective):
        return objective.raw_input
    evaluator = getattr(objective, '__self__', None)
    if not isinstance(evaluator, PipelineObjectiveEvaluateWithTensorData):
        return None
    producer = evaluator._data_producer
    if not isinstance(producer, partial) or not producer.args:
        return None
    data = evaluator.tensor_data
    if not isinstance(data, TensorData):
        return None
    data_type = getattr(data.data_type, 'name', data.data_type)
    if data_type == 'table':
        data_type = 'tabular'
    if data_type not in ('tabular', 'image', 'ts'):
        return None
    return RawInputSpec(data_type, has_target=data.target is not None)

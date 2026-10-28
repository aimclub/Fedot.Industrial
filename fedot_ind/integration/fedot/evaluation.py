"""Preserve FEDOT's typed failure at the Industrial worker boundary."""

from functools import partial

from fedot.core.data.tensor_data import TensorData
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
from fedot.core.optimisers.objective.evaluation_contracts import EvaluationIncomplete

from fedot_ind.core.optimizer.domain import EvaluationFailure, EvaluationFailureCode
from fedot_ind.core.optimizer.mutation_planning import RawInputSpec


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

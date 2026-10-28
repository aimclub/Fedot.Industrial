from functools import partial

import numpy as np
import pytest

from fedot import create_data
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData

from fedot.core.optimisers.objective.evaluation_contracts import (
    EvaluationFailure, EvaluationIncomplete, FailureKind,
)

from fedot_ind.integration.fedot.evaluation import confirmed_input_from_objective, industrial_failure_from_fedot
from fedot_ind.core.optimizer.domain import EvaluationFailureCode
from fedot_ind.core.optimizer.mutation_planning import RawInputSpec


@pytest.mark.parametrize('kind', list(FailureKind))
@pytest.mark.parametrize('error_type', ['', 'TypeError'])
def test_native_failure_preserves_category_message_and_exception(kind, error_type):
    outcome = EvaluationIncomplete(
        'candidate', 0, (), (), EvaluationFailure(kind, 'original explanation', error_type))

    translated = industrial_failure_from_fedot(outcome)

    assert translated.code is EvaluationFailureCode.OBJECTIVE_INCOMPLETE
    assert translated.error_type == (error_type or f'FedotEvaluation:{kind.value}')
    assert translated.message == f'{kind.value}: original explanation'


@pytest.mark.parametrize('outcome', [None, {}, 'failure'])
def test_unknown_outcomes_are_not_misclassified(outcome):
    assert industrial_failure_from_fedot(outcome) is None


@pytest.mark.parametrize('data_type, normalized_type, shape', [
    ('tabular', 'tabular', (4, 3)), ('image', 'ts', (4, 1, 8)),
])
@pytest.mark.parametrize('has_target', [True, False])
def test_confirmed_input_uses_existing_tensor_source_without_executing_producer(
        data_type, normalized_type, shape, has_target):
    data = create_data(np.ones(shape), target=np.arange(4) if has_target else None,
                       task='regression', data_type=data_type, without_target=not has_target)
    evaluator = object.__new__(PipelineObjectiveEvaluateWithTensorData)
    evaluator._data_producer = partial(lambda data: pytest.fail('producer was consumed'), data)

    assert confirmed_input_from_objective(evaluator.evaluate) == RawInputSpec(normalized_type, has_target)


def test_custom_source_is_not_consumed_or_guessed():
    evaluator = object.__new__(PipelineObjectiveEvaluateWithTensorData)
    evaluator._data_producer = lambda: pytest.fail('custom producer was consumed')

    assert confirmed_input_from_objective(evaluator.evaluate) is None
    assert confirmed_input_from_objective(lambda graph: None) is None


def test_non_tensor_or_missing_partial_argument_does_not_confirm_input():
    evaluator = object.__new__(PipelineObjectiveEvaluateWithTensorData)
    evaluator._data_producer = partial(lambda data: (), {'modality': 'data'})
    assert confirmed_input_from_objective(evaluator.evaluate) is None
    evaluator._data_producer = partial(lambda: ())
    assert confirmed_input_from_objective(evaluator.evaluate) is None

from dataclasses import FrozenInstanceError
import json
import pickle

import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.evaluation.context import (
    EvaluationContext, current_evaluation_context, evaluation_context_scope,
)
from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json


_JSON = st.recursive(st.none() | st.booleans() | st.integers() | st.text() |
                     st.floats(allow_infinity=False, allow_nan=False),
                     lambda child: st.lists(child, max_size=4) | st.dictionaries(st.text(), child, max_size=4),
                     max_leaves=15)


@given(st.dictionaries(st.text(), _JSON, max_size=6))
def test_canonical_json_ignores_mapping_order_and_round_trips(values):
    reversed_values = dict(reversed(list(values.items())))
    actual = canonical_json(values)
    assert actual == canonical_json(reversed_values)
    assert canonical_json(json.loads(actual)) == actual
    assert PreparedEvaluationGraph('graph-a', actual).configuration_id == (
        PreparedEvaluationGraph('graph-b', canonical_json(reversed_values)).configuration_id)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {1: 'bad'}, {True: 1}, (1, 2), object()])
def test_non_json_configuration_values_are_not_stringified(value):
    with pytest.raises(ValueError):
        canonical_json(value)


def test_configuration_snapshot_is_immutable_and_array_order_is_semantic():
    first = PreparedEvaluationGraph('graph', canonical_json({'branches': [1, 2]}))
    second = PreparedEvaluationGraph('graph', canonical_json({'branches': [2, 1]}))
    assert first.configuration_id != second.configuration_id
    with pytest.raises(FrozenInstanceError):
        first.configuration_json = '{}'
    with pytest.raises(ValueError, match='canonical'):
        PreparedEvaluationGraph('graph', '{"b": 1, "a": 2}')


def test_evaluation_context_is_pickleable_and_scoped_on_success_and_failure():
    context = EvaluationContext('candidate:0', 'candidate', 'config', 0, canonical_json({'x': 1}))
    assert pickle.loads(pickle.dumps(context)) == context
    assert current_evaluation_context() is None
    with evaluation_context_scope(context):
        assert current_evaluation_context() == context
        with pytest.raises(ValueError):
            with evaluation_context_scope(None):
                assert current_evaluation_context() is None
                raise ValueError('failure')
        assert current_evaluation_context() == context
    assert current_evaluation_context() is None
    assert context.to_record()['configuration'] == {'x': 1}

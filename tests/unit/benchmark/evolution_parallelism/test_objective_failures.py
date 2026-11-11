from types import SimpleNamespace
import json

import pytest
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.fitness import null_fitness
from fedot.core.optimisers.objective.evaluation_contracts import EvaluationIncomplete, EvaluationFailure, FailureKind
from fedot.core.operations.operation import Operation

from benchmark.industrial.experiments.evolution_parallelism.objective import SearchObjective
from benchmark.industrial.experiments.evolution_parallelism.operation_timing import operation_trace, exception_chain
from benchmark.industrial.experiments.evolution_parallelism.search import MeasuredObjective, loss_value
from fedot_ind.integration.fedot.evaluation import ConfirmedObjective, objective_outcome
from fedot_ind.core.optimizer.mutation_planning import RawInputSpec
from fedot_ind.core.optimizer.evaluation.dispatcher import _GraphEvaluator, _GraphJob


class NativeFailure:
    def evaluate(self, graph):
        self.last_outcome = EvaluationIncomplete('g', 3, (), (),
                                                 EvaluationFailure(FailureKind.FIT, 'actual failure', 'ExtensionContractError'))
        return null_fitness()


def test_wrappers_preserve_native_failure_through_worker_boundary(tmp_path):
    native = NativeFailure()
    wrapped = ConfirmedObjective(MeasuredObjective(native, str(tmp_path)), RawInputSpec('ts', True))
    result = _GraphEvaluator(DirectAdapter(), wrapped, None, None)(_GraphJob(OptGraph([OptNode('ridge')]), 'g'))
    assert result.metadata['evaluation_error_type'] == 'ExtensionContractError'
    assert result.metadata['evaluation_error'] == 'fit: actual failure'
    assert objective_outcome(wrapped) is native.last_outcome
    assert objective_outcome(native.evaluate) is native.last_outcome


def test_strict_metric_does_not_turn_original_exception_into_null_fitness():
    def metric(graph):
        raise ValueError('prediction shape mismatch')
    with pytest.raises(ValueError, match='shape mismatch'):
        SearchObjective(metric).evaluate_strict(None)


def test_operation_trace_preserves_nested_cause_and_restores_original(tmp_path, monkeypatch):
    def fail(self, *args):
        try:
            raise ValueError('root cause')
        except ValueError as error:
            raise RuntimeError('wrapper') from error
    monkeypatch.setattr(Operation, 'fit', fail)
    with pytest.raises(RuntimeError, match='wrapper'):
        with operation_trace(tmp_path, {'evaluation_id': 'g:1'}):
            Operation.fit(SimpleNamespace(operation_type='fourier_basis'))
    assert Operation.fit is fail
    row = json.loads(next(tmp_path.glob('operations_*.jsonl')).read_text().strip())
    assert row['status'] == 'failed'
    assert row['error_chain'] == [{'type': 'RuntimeError', 'message': 'wrapper'},
                                  {'type': 'ValueError', 'message': 'root cause'}]


def test_exception_chain_is_bounded_for_a_cycle():
    a, b = ValueError('a'), RuntimeError('b')
    a.__cause__, b.__cause__ = b, a
    assert len(exception_chain(a)) == 2


def test_finite_float32_predictions_do_not_overflow_the_rmse_square():
    import numpy as np
    target = np.array([0., 0.], dtype=np.float32)
    prediction = np.array([1e20, -1e20], dtype=np.float32)
    result = loss_value('regression', target, prediction)
    assert np.isfinite(result)
    assert result == pytest.approx(float(prediction[0]))


def test_rmse_remains_finite_when_even_a_float64_square_would_overflow():
    import numpy as np
    assert loss_value('regression', np.zeros(2), np.array([1e200, -1e200])) == pytest.approx(1e200)

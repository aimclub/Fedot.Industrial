"""Real composite pipelines preserve features, predictions and node ownership."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from benchmark.industrial.experiments.mutation_planning.population_study.supervised_diagnostics import (
    _pipeline, _runtime, _tensor,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import (
    feature_reuse_scope, reuse_statistical_features,
)
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor


PARAMS = {'quantile_extractor': {'use_cache': False, 'use_sliding_window': False, 'window_size': 0},
          'quantile_extractor_torch': {'use_cache': False, 'use_sliding_window': False, 'window_size': 0},
          'logit': {'random_state': 42}, 'ridge': {'alpha': 1.}}


@pytest.fixture(autouse=True)
def deterministic_torch():
    enabled, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn)


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_actual_branches_have_exact_features_and_predictions_with_one_less_computation(task):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(20, 1, 48))
    target = np.arange(20) % 2 if task == 'classification' else rng.normal(size=20)
    train, test = _tensor(x, target, task), _tensor(x + .2, None, task)
    original = deepcopy(train), deepcopy(test)
    outputs, counts = [], []
    with _runtime():
        for budget in (0, 32 * 1024 * 1024):
            pipeline = _pipeline('composite', task, PARAMS)
            calls = []

            def counted(features, settings, compute):
                def execute():
                    calls.append(1)
                    return compute()
                return reuse_statistical_features(features, settings, execute)

            module = 'fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor'
            with patch(f'{module}.reuse_statistical_features', counted):
                with feature_reuse_scope(max_bytes=budget) as fitting:
                    fitted = pipeline.fit(train)
                fit_count = len(calls)
                with feature_reuse_scope(max_bytes=budget) as predicting:
                    predicted = pipeline.predict(test)
                counts.append((fit_count, len(calls) - fit_count))
            outputs.append((fitted, predicted))
            nodes = [n for n in pipeline.nodes if n.name == 'quantile_extractor']
            assert len({id(n.fitted_operation) for n in nodes}) == len(nodes)
            if budget:
                assert fitting.statistics.hits == predicting.statistics.hits == 1
            pipeline.unfit()
    assert counts == [(2, 2), (1, 1)]
    for old, new in zip(*outputs):
        assert torch.equal(old.features, new.features)
        assert torch.equal(old.predict, new.predict)
    assert train == original[0] and test == original[1]


def test_effective_parameters_and_filtering_are_not_shared():
    from types import SimpleNamespace
    x = np.random.default_rng(12).normal(size=(5, 1, 32))
    data = SimpleNamespace(features=x)
    params = {'use_cache': False, 'window_size': 0, 'use_sliding_window': False}
    first, second = [QuantileExtractor(OperationParameters(**params)) for _ in range(2)]
    with _runtime(), feature_reuse_scope() as session:
        first._transform(data)
        second._transform(data)
        second.add_global_features = False
        second._transform(data)
        assert session.statistics.hits == 1 and session.statistics.misses == 2
        second.feature_filter = object()  # Even disabled but externally owned filters bypass reuse.
        second._transform(data)
        assert session.statistics.hits == 1 and session.statistics.misses == 2
    assert first.predict is not second.predict


@pytest.mark.parametrize('channel', [True, False])
@pytest.mark.parametrize('attribute,value', [('window_size', 20), ('stride', 2),
                                             ('add_global_features', False), ('use_sliding_window', True)])
def test_effective_setting_changes_match_independent_reference(channel, attribute, value):
    from types import SimpleNamespace
    x = np.random.default_rng(14).normal(size=(4, 1, 32))
    params = {'use_cache': False, 'use_sliding_window': False, 'channel_independent': channel}
    a, b, reference = [QuantileExtractor(OperationParameters(**params)) for _ in range(3)]
    setattr(b, attribute, value)
    setattr(reference, attribute, value)
    data = SimpleNamespace(features=x)
    with _runtime():
        expected = reference._transform(data)
        with feature_reuse_scope() as session:
            a._transform(data)
            actual = b._transform(data)
            assert session.statistics.misses == 2
        np.testing.assert_array_equal(expected, actual)


def test_actual_channel_mode_is_part_of_the_key():
    from types import SimpleNamespace
    data = SimpleNamespace(features=np.random.default_rng(17).normal(size=(4, 1, 32)))
    extractor = QuantileExtractor(OperationParameters(use_cache=False))
    with _runtime(), feature_reuse_scope() as session:
        rowwise = extractor._transform(data)
        extractor.channel_extraction = False
        batched = extractor._transform(data)
        assert session.statistics.misses == 2 and session.statistics.hits == 0
        assert rowwise.shape != batched.shape


def test_custom_instance_and_subclass_are_never_memoized():
    from types import MethodType, SimpleNamespace

    class CustomQuantile(QuantileExtractor):
        pass

    data = SimpleNamespace(features=np.random.default_rng(2).normal(size=(3, 1, 32)))
    params = OperationParameters(use_cache=False)
    with _runtime(), feature_reuse_scope() as session:
        for obj in (QuantileExtractor(params), CustomQuantile(params)):
            calls = []

            def custom(self, values):
                calls.append(1)
                return np.zeros((3, 1, 2))

            obj._clean_predict = MethodType(custom, obj)
            obj._transform(data)
            obj._transform(data)
            assert len(calls) == 2
        assert session.statistics.hits == session.statistics.misses == 0


def test_identical_features_do_not_share_targets_or_fitted_models():
    x = np.random.default_rng(21).normal(size=(20, 1, 32))
    target = np.arange(20) % 2
    with _runtime(), feature_reuse_scope() as session:
        a, b = [_pipeline('simple', 'classification', PARAMS) for _ in range(2)]
        first = a.fit(_tensor(x, target, 'classification'))
        second = b.fit(_tensor(x, 1 - target, 'classification'))
        assert torch.equal(first.features, second.features)
        assert not torch.equal(first.target, second.target)
        assert a.root_node.fitted_operation is not b.root_node.fitted_operation
        assert not torch.equal(first.predict, second.predict)
        assert session.statistics.hits == session.statistics.misses == 1
        a.unfit()
        b.unfit()


@pytest.mark.parametrize('backend', ['local', 'legacy'])
def test_dispatchers_own_reuse_storage_per_candidate(backend):
    from types import SimpleNamespace
    from golem.core.optimisers.fitness import SingleObjFitness
    from fedot_ind.core.optimizer.evaluation.dispatcher import _GraphEvaluator, _GraphJob
    from fedot_ind.core.repository.IndustrialDispatcher import IndustrialDispatcher

    calls = []
    features = np.ones(2)

    def compute():
        calls.append(1)
        return features.copy()

    def objective(graph):
        a = reuse_statistical_features(features, (), compute)
        b = reuse_statistical_features(features, (), compute)
        return SingleObjFitness(float((a + b).sum()))

    adapter = SimpleNamespace(adapt_func=lambda function: function)
    if backend == 'local':
        evaluator = _GraphEvaluator(adapter, objective, None, None)

        def evaluate(index):
            return evaluator(_GraphJob('graph', str(index)))
    else:
        dispatcher = object.__new__(IndustrialDispatcher)
        dispatcher._adapter = adapter
        dispatcher._evaluate_graph = lambda graph: (objective(graph), graph)

        def evaluate(index):
            return dispatcher.eval_ind('graph', str(index)).compute(scheduler='synchronous')
    for index in range(2):
        assert evaluate(index).fitness.valid
    assert len(calls) == 2
    reuse_statistical_features(features, (), compute)
    reuse_statistical_features(features, (), compute)
    assert len(calls) == 4  # No scope, no reuse after either dispatch.

"""Forecast merge boundaries, without model training or dataset access."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.models.ts_forecasting.branching.contracts import ForecastBranchError, ForecastMergeKind
from fedot_ind.core.models.ts_forecasting.branching.operations import (
    CenterWindows, DifferenceWindows, SmoothWindows, MeanForecast)
from fedot_ind.integration.fedot.forecast_branches import merge_forecast_branches


def branch(offset=0., prediction=False):
    values = torch.arange(12, dtype=torch.float64).reshape(4, 3) + offset
    return TensorData(features=values, target=torch.arange(8, dtype=torch.float64).reshape(4, 2),
                      idx=np.array([5, 2, 8, 4]), task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(2)),
                      data_type=DataTypesEnum.table, predict=values[:, :2].clone() if prediction else None)


@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('kind', list(ForecastMergeKind))
def test_merge_conserves_order_targets_and_each_branch(count, kind):
    outputs = [branch(float(i), kind is ForecastMergeKind.PREDICTIONS) for i in range(count)]
    before = outputs[0].features.clone()
    result = merge_forecast_branches(outputs, kind)
    field = 'predict' if kind is ForecastMergeKind.PREDICTIONS else 'features'
    torch.testing.assert_close(result.features, torch.cat([getattr(o, field) for o in outputs], dim=1))
    np.testing.assert_array_equal(result.idx, [5, 2, 8, 4])
    torch.testing.assert_close(result.target, outputs[0].target)
    torch.testing.assert_close(outputs[0].features, before)
    assert result.predict is None


@pytest.mark.parametrize('change', ['order', 'target', 'horizon', 'missing_target', 'dtype', 'nan', 'duplicate_idx'])
def test_feature_merge_rejects_incompatible_alignment(change):
    first, second = branch(), branch()
    changes = {
        'order': {'idx': second.idx[::-1]}, 'target': {'target': second.target + 1},
        'horizon': {'task': Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(3))},
        'missing_target': {'target': None}, 'dtype': {'features': second.features.float()},
        'nan': {'features': torch.full_like(second.features, float('nan'))},
        'duplicate_idx': {'idx': np.array([5, 5, 8, 4])}}
    second = replace(second, **changes[change])
    with pytest.raises(ForecastBranchError):
        merge_forecast_branches([first, second], ForecastMergeKind.FEATURES)


def test_prediction_merge_cannot_fall_back_to_features_or_partial_forecasts():
    with pytest.raises(ForecastBranchError):
        merge_forecast_branches([branch(), branch()], ForecastMergeKind.PREDICTIONS)
    with pytest.raises(ForecastBranchError):
        merge_forecast_branches([branch(prediction=True), replace(branch(prediction=True), predict=torch.ones(4, 1))],
                                ForecastMergeKind.PREDICTIONS)


@pytest.mark.parametrize('kind', list(ForecastMergeKind))
def test_horizon_can_be_explicit_in_tensor_data_without_task_parameters(kind):
    outputs = [replace(branch(float(i), kind is ForecastMergeKind.PREDICTIONS),
                       task=Task(TaskTypesEnum.ts_forecasting), ts_forecast_horizon=2) for i in range(2)]
    result = merge_forecast_branches(outputs, kind)
    assert result.ts_forecast_horizon == 2
    with pytest.raises(ForecastBranchError, match='horizon'):
        merge_forecast_branches([outputs[0], replace(outputs[1], ts_forecast_horizon=3)], kind)


@pytest.mark.parametrize('count', [2, 3, 4])
def test_forecast_mean_matches_independent_arithmetic_reference(count):
    outputs = [branch(float(i), True) for i in range(count)]
    merged = merge_forecast_branches(outputs, ForecastMergeKind.PREDICTIONS)
    data = SimpleNamespace(features=merged.features.numpy(), task=merged.task)
    mean = MeanForecast().fit(data)
    expected = np.mean([o.predict.numpy() for o in outputs], axis=0)
    np.testing.assert_allclose(mean.predict(data), expected)
    # Blending does not fit weights to any targets.
    np.testing.assert_allclose(mean.predict(SimpleNamespace(
        features=data.features, task=data.task, target=1e20)), expected)


@pytest.mark.parametrize('implementation', [CenterWindows, DifferenceWindows, SmoothWindows])
def test_representations_do_not_mix_samples_and_do_not_mutate_windows(implementation):
    values = np.arange(40.).reshape(5, 8)
    before = values.copy()
    actual = implementation().transform(values)
    expected = np.concatenate([implementation().transform(row[None, :]) for row in values])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, before)
    assert actual.shape == values.shape


def test_plain_fedot_factory_refuses_an_ambiguous_prediction_merge():
    from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
    from fedot_ind.integration.fedot.extensions.factories import make_deferred_factory
    declaration = load_industrial_extension_catalog().operation('forecast_mean')
    model = make_deferred_factory(declaration)({'forecast_length': 2})
    with pytest.raises(ForecastBranchError, match='ForecastPipelineAdapter'):
        model.fit(np.ones((4, 4)))


def test_default_discovery_does_not_enable_adapter_specific_operations():
    from fedot_ind.integration.fedot.extensions.discovery import catalog_operation_names
    assert not {'forecast_mean', 'forecast_feature_join', 'forecast_window_center',
                'forecast_window_difference', 'forecast_window_smooth'}.intersection(catalog_operation_names('ts_forecasting'))


def test_stateless_transform_factory_does_not_require_a_fit_implementation():
    from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
    from fedot_ind.integration.fedot.extensions.factories import make_deferred_factory
    declaration = load_industrial_extension_catalog().operation('forecast_window_center')
    runtime = make_deferred_factory(declaration)({})
    values = np.array([[1., 2., 3.]])
    assert runtime.fit(values) is runtime
    np.testing.assert_allclose(runtime.transform(values), [[-1., 0., 1.]])


@pytest.mark.parametrize('mode', ['fit', 'predict'])
def test_merge_context_is_restored_when_node_execution_fails(monkeypatch, mode):
    from fedot.core.pipelines.node import PipelineNode
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from fedot_ind.integration.fedot.forecast_branches import ForecastMergeNode
    from fedot_ind.integration.fedot.forecast_merge_context import active_forecast_merge

    def failure(*args, **kwargs):
        active_forecast_merge.set(ForecastMergeKind.PREDICTIONS)
        raise RuntimeError('Simulated downstream failure')

    monkeypatch.setattr(PipelineNode, mode, failure)
    with industrial_extension_scope():
        node = ForecastMergeNode('forecast_mean')
        assert active_forecast_merge.get() is None
        with pytest.raises(RuntimeError, match='Simulated'):
            getattr(node, mode)(branch())
        assert active_forecast_merge.get() is None


def test_window_representations_match_independent_formulas():
    values = np.array([[1., 3., 6., 10.], [10., 6., 3., 1.]])
    np.testing.assert_array_equal(CenterWindows().transform(values), values - 5.)
    np.testing.assert_array_equal(DifferenceWindows().transform(values), [[0., 2., 3., 4.], [0., -4., -3., -2.]])
    np.testing.assert_array_equal(SmoothWindows().transform(values), [[1., 2., 4.5, 8.], [10., 8., 4.5, 2.]])


def test_runtime_catalog_requires_the_explicit_adapter():
    from benchmark.industrial.experiments.evolution_parallelism.search import build_environment
    from fedot_ind.core.optimizer.mutation_runtime import load_runtime_operation_catalog
    from fedot_ind.integration.fedot.forecast_branches import ForecastPipelineAdapter
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    settings = {'tasks': {'ts_forecasting': {'operations': ['hankelisation', 'ridge', 'forecast_mean']}},
                'max_depth': 6, 'max_arity': 4, 'budget_seconds': 30}
    with industrial_extension_scope():
        _, params = build_environment(settings, branch().task, 1)
        ordinary = load_runtime_operation_catalog(params)
        assert 'forecast_mean' in ordinary.unavailable
        assert 'forecast_mean' not in {op.name for op in ordinary.capabilities}
        params.adapter = ForecastPipelineAdapter()
        explicit = load_runtime_operation_catalog(params)
        assert 'forecast_mean' in {op.name for op in explicit.capabilities}


@pytest.mark.parametrize('merge', ['forecast_feature_join', 'forecast_mean'])
def test_mutation_interpreter_advisor_and_native_verifier_accept_two_to_four_branches(merge):
    from benchmark.industrial.experiments.evolution_parallelism.search import build_environment
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.verification import verifier_for_task
    from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier, ModelFeatureFlow
    from fedot_ind.core.optimizer.mutation_planning import (
        AppliedMutation, ForkForecastBranch, MutationLimits, MutationPlan, RawInputSpec,
        apply_mutation_plan, graph_signature)
    from fedot_ind.core.optimizer.mutation_runtime import (
        advisor_allows_plan, install_extension_aware_advisor, interpret_graph_spec,
        load_runtime_operation_catalog, project_for_mutation)
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from fedot_ind.integration.fedot.forecast_branches import ForecastPipelineAdapter
    settings = {'tasks': {'ts_forecasting': {'operations': [
        'hankelisation', 'ridge', 'forecast_mean', 'forecast_feature_join', 'forecast_window_center',
        'forecast_window_difference', 'forecast_window_smooth']}},
        'max_depth': 6, 'max_arity': 4, 'budget_seconds': 30}
    raw, limits = RawInputSpec('ts', True), MutationLimits(16, 6, max_arity=4)
    with industrial_extension_scope():
        _, params = build_environment(settings, branch().task, 1)
        params.adapter = ForecastPipelineAdapter()
        params.verifier = IndustrialGraphVerifier(
            adapter=params.adapter, task_type='ts_forecasting', input_data_type='ts',
            model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES,
            legacy_verifier=verifier_for_task(TaskTypesEnum.ts_forecasting, params.adapter))
        install_extension_aware_advisor(params)
        graph = params.adapter.adapt(PipelineBuilder().add_node('hankelisation').add_node('ridge').build())
        catalog = {op.name: op for op in load_runtime_operation_catalog(params).capabilities}
        for count, name in enumerate(['forecast_window_center', 'forecast_window_difference', 'forecast_window_smooth'], 2):
            before = project_for_mutation(graph, params)
            child = next((n for n in before.nodes if n.operation_name == merge),
                         next(n for n in before.nodes if n.operation_name == 'ridge'))
            window = next(n for n in before.nodes if n.operation_name == 'hankelisation')
            operations = (catalog[name],) + ((catalog['ridge'],) if merge == 'forecast_mean' else ())
            edit = ForkForecastBranch(child.node_id, window.node_id, operations, catalog[merge])
            outcome = apply_mutation_plan(before, MutationPlan(graph_signature(before), (edit,), raw),
                                          limits=limits, raw_input=raw)
            assert isinstance(outcome, AppliedMutation), outcome
            assert advisor_allows_plan(before, outcome.graph, graph, params.advisor)
            graph = interpret_graph_spec(graph, before, outcome.graph)
            assert params.verifier(graph), params.verifier.last_report
            restored = project_for_mutation(graph, params)
            assert len(next(n for n in restored.nodes if n.operation_name == merge).parent_ids) == count


def test_adapter_restoration_and_prediction_route_use_predict_not_features(monkeypatch):
    from fedot.core.operations.operation import Operation
    from golem.core.optimisers.graph import OptGraph, OptNode
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from fedot_ind.integration.fedot.forecast_branches import ForecastMergeNode, ForecastPipelineAdapter
    from fedot_ind.integration.fedot.forecast_merge_context import active_forecast_merge
    import fedot_ind.integration.fedot.forecast_branches as runtime
    with industrial_extension_scope():
        first, second = OptNode('ridge'), OptNode('ridge')
        root = OptNode({'name': 'forecast_mean', 'params': {'forecast_length': 2}}, nodes_from=[first, second])
        pipeline = ForecastPipelineAdapter().restore(OptGraph(root))
        assert isinstance(pipeline.root_node, ForecastMergeNode)
        outputs = [branch(0., True), branch(10., True)]
        monkeypatch.setattr(runtime, '_combine_parents_tensordata', lambda *args, **kwargs: outputs)
        with patch.object(Operation, 'fit', side_effect=AssertionError('Training forbidden')) as fit:
            token = active_forecast_merge.set(None)
            try:
                merged = pipeline.root_node._tensordata_from_parents(branch(), 'predict')
                assert active_forecast_merge.get() is ForecastMergeKind.PREDICTIONS
                assert merged.features.shape == (4, 4)
                assert fit.call_count == 0
            finally:
                active_forecast_merge.reset(token)


@pytest.mark.parametrize('horizon', [1, 3])
@pytest.mark.parametrize('merge', ['forecast_feature_join', 'forecast_mean'])
def test_public_pipeline_routes_window_features_and_predictions_without_training(monkeypatch, horizon, merge):
    """Exercise native fit/predict orchestration with a non-learning Ridge stand-in."""
    from sklearn.linear_model import Ridge
    from examples.release_1_0.evolution.forecast_branches import build_branch_graphs, restore_branch_graph
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from fedot_ind.integration.fedot.forecast_merge_context import active_forecast_merge

    calls = []

    def no_learning_fit(model, features, target, **kwargs):
        # Coefficients are fixed in advance, not estimated from features or labels.
        values = np.asarray(features)
        model.n_features_in_ = values.shape[1]
        model.coef_ = np.full((horizon, values.shape[1]), 1. / values.shape[1])
        model.intercept_ = np.arange(horizon, dtype=float)
        calls.append(values.shape)
        return model

    monkeypatch.setattr(Ridge, 'fit', no_learning_fit)
    values = np.sin(np.arange(60.) * .2)
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon))
    history = torch.tensor(values[None, None, :], dtype=torch.float64)
    data = TensorData(features=history, target=history.clone(), task=task, ts_forecast_horizon=horizon,
                      idx=np.array([0]), data_type=DataTypesEnum.ts)
    with industrial_extension_scope():
        settings, graphs = build_branch_graphs()
        settings['forecast_length'] = horizon
        graph = next(g for name, count, g in graphs if name == merge and count == 4)
        pipeline = restore_branch_graph(graph, settings)
        fitted = pipeline.fit(data)
        assert fitted.predict.shape[-1] == horizon
        assert active_forecast_merge.get() is None
        test = replace(data, target=None, state='predict')
        output = pipeline.predict(test)
        assert output.predict.numel() == horizon
        assert torch.isfinite(output.predict).all()
        assert active_forecast_merge.get() is None
        assert len(calls) == (1 if merge == 'forecast_feature_join' else 4)
        window = values[-settings['window_size']:]
        branch_means = [window.mean(), 0., np.diff(window, prepend=window[0]).mean(),
                        ((window + np.r_[window[0], window[:-1]]) / 2.).mean()]
        np.testing.assert_allclose(output.predict.numpy().reshape(-1), np.mean(branch_means) + np.arange(horizon),
                                   atol=1e-7)

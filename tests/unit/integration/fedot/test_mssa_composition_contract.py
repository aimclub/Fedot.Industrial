"""SSA-family forecasts are terminal outputs, not aligned features for Ridge."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from fedot import create_data
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.models.ts_forecasting.lagged_model.mssa_forecaster import MSSAForecaster
from fedot_ind.core.optimizer.graph_validation import (
    IndustrialGraphVerifier, NodePosition, ValidationIssueCode, graph_spec_from_runtime,
    industrial_capability_index, validate_graph,
)
from fedot_ind.core.optimizer.mutation_planning import (
    AppliedMutation, MutationKind, MutationLimits, RawInputSpec, apply_mutation_plan,
    enumerate_mutation_plans,
)
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


def _pipeline(*operations):
    builder = PipelineBuilder()
    for name in operations:
        params = ({'device': 'cpu', 'head_policy': 'linear', 'window_size': 12, 'rank': 2}
                  if name in ('mssa_forecaster', 'ssa_forecaster') else None)
        if name == 'ssa_forecaster':
            params['history_lookback'] = 0
        builder.add_node(name, params=params)
    return builder.build()


@pytest.mark.parametrize('model', ['mssa_forecaster', 'ssa_forecaster'])
def test_ssa_family_rejects_forecast_as_training_features_before_fitting(model):
    assert industrial_capability_index()[model].allowed_positions == (NodePosition.ROOT,)
    with industrial_extension_scope():
        invalid = _pipeline(model, 'ridge')
        verifier = IndustrialGraphVerifier(adapter=None, task_type='ts_forecasting')
        report = verifier.verify_with_report(invalid)
        assert not report.is_valid
        issue, = [issue for issue in report.issues if issue.code is ValidationIssueCode.POSITION_NOT_ALLOWED]
        assert issue.operation_names == (model,)
        assert all(node.fitted_operation is None for node in invalid.nodes)
        assert verifier.verify(_pipeline(model))
        assert verifier.verify(_pipeline('smoothing', model))
        assert verifier.verify(_pipeline('hankelisation', 'ridge'))


@pytest.mark.parametrize('model', ['mssa_forecaster', 'ssa_forecaster'])
def test_planner_cannot_place_terminal_forecaster_before_a_downstream_model(model):
    with industrial_extension_scope():
        spec = graph_spec_from_runtime(_pipeline(model, 'ridge'),
                                       task_type='ts_forecasting')
        # A pure old-contract control isolates the position rule without allowing
        # an invalid actual SSA pipeline to become an executable reference.
        spec = replace(spec, nodes=tuple(
            replace(node, operation_name='unrestricted_forecaster', capabilities=replace(
                node.capabilities, name='unrestricted_forecaster', allowed_positions=(NodePosition.ANY,)))
            if node.operation_name == model else node for node in spec.nodes))
        assert validate_graph(spec).is_valid
        forecast = industrial_capability_index()[model]
        limits = MutationLimits(max_nodes=6, max_depth=6, max_candidates=128, max_arity=2)
        raw = RawInputSpec('ts', True)
        candidates = enumerate_mutation_plans(spec, (forecast,), limits=limits,
                                              kinds=(MutationKind.REPLACE,), raw_input=raw)
        assert not candidates.plans
        assert any(ValidationIssueCode.POSITION_NOT_ALLOWED.value in item.reason_codes
                   for item in candidates.diagnostics)
        other = 'ssa_forecaster' if model == 'mssa_forecaster' else 'mssa_forecaster'
        terminal = graph_spec_from_runtime(_pipeline('smoothing', other),
                                           task_type='ts_forecasting')
        candidates = enumerate_mutation_plans(terminal, (forecast,), limits=limits,
                                              kinds=(MutationKind.REPLACE,), raw_input=raw)
        assert candidates.plans
        for plan in candidates.plans:
            result = apply_mutation_plan(terminal, plan, limits=limits, raw_input=raw)
            assert isinstance(result, AppliedMutation)
            forecast_node, = [node for node in result.graph.nodes if node.operation_name == model]
            assert not any(forecast_node.node_id in node.parent_ids for node in result.graph.nodes)


@pytest.mark.parametrize('planned', [False, True])
@pytest.mark.parametrize('model', ['mssa_forecaster', 'ssa_forecaster'])
def test_both_optimizer_modes_enforce_terminal_forecaster_without_training(planned, model):
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.objective import Objective
    from benchmark.industrial.experiments.mutation_planning.runner import _environment
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy

    with industrial_extension_scope():
        requirements, params, graph = _environment(
            {'name': 'ssa-boundary', 'task': 'ts_forecasting',
             'operations': [model, 'ridge', 'hankelisation'],
             'initial_chain': [model]}, {'max_depth': 4, 'max_arity': 2})
        optimizer = IndustrialEvoOptimizer(
            Objective({'size': lambda pipeline: len(pipeline.nodes)}), [graph], requirements,
            params, GPAlgorithmParameters(pop_size=2),
            EvolutionConfig(random_seed=42, mutation_planning=MutationPlanningPolicy(enabled=planned)))
        invalid = _pipeline(model, 'ridge')
        report = optimizer.graph_verifier.verify_with_report(params.adapter.adapt(invalid))
        assert ValidationIssueCode.POSITION_NOT_ALLOWED.value in report.error_codes
        assert optimizer.graph_verifier.verify(graph)
        assert all(node.fitted_operation is None for node in invalid.nodes)


@pytest.mark.parametrize('horizon', [1, 7, 14])
@pytest.mark.parametrize('model', ['mssa_forecaster', 'ssa_forecaster'])
def test_terminal_forecaster_matches_independent_core_without_prediction_targets(horizon, model):
    history = (np.sin(np.arange(120, dtype=float) / 4) + np.arange(120) * 0.02).astype(np.float32)
    data = create_data(history, target=history,
                       task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon)),
                       ts_forecast_horizon=horizon, without_target=True, use_cache=False)
    predict_data = replace(data, target=None, state='predict')
    snapshots = data.features.clone(), data.target.clone(), data.idx.clone()
    expected = MSSAForecaster(forecast_horizon=horizon, window_size=12, rank=2,
                              head_policy='linear', device='cpu').fit(history).predict(history)
    with industrial_extension_scope():
        pipeline = _pipeline(model)
        try:
            pipeline.fit(data)
            actual = pipeline.predict(predict_data)
            assert actual.predict.shape == (horizon,)
            assert torch.isfinite(actual.predict).all()
            np.testing.assert_allclose(actual.predict.numpy(), expected, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(data.features, snapshots[0])
            torch.testing.assert_close(data.target, snapshots[1])
            np.testing.assert_array_equal(data.idx, snapshots[2])
            assert predict_data.target is None
        finally:
            pipeline.unfit()

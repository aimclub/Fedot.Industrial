"""Confirmed TensorData inputs constrain both evolutionary mutation modes."""

from functools import partial
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fedot import create_data
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer, IndustrialPopulationError
from fedot_ind.core.optimizer.graph_validation import (
    IndustrialGraphVerifier, ModelFeatureFlow, graph_spec_from_runtime,
)
from fedot_ind.core.optimizer.mutation_planning import RawInputSpec
from fedot_ind.core.optimizer.mutation_runtime import PlannedMutation
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


def _pipeline(*chain):
    builder = PipelineBuilder()
    for name in chain:
        builder.add_node(name)
    return builder.build()


def _objective(task='classification'):
    data = create_data(np.ones((8, 1, 16)), target=np.arange(8) % 2,
                       task=task, data_type='ts', use_cache=False)
    evaluator = object.__new__(PipelineObjectiveEvaluateWithTensorData)
    evaluator._data_producer = partial(lambda data: pytest.fail('producer consumed during admission'), data)
    return evaluator.evaluate


@pytest.mark.parametrize('planned', [False, True])
@pytest.mark.parametrize('task,chain', [
    ('classification', ('logit',)), ('classification', ('dt',)),
    ('regression', ('ridge',)), ('regression', ('dtreg',)),
    ('regression', ('ridge', 'pdl_reg')), ('classification', ('pdl_clf', 'logit')),
])
def test_optimizer_admits_no_training_for_all_recorded_shape_failure_families(planned, task, chain):
    with industrial_extension_scope():
        optimizer = object.__new__(IndustrialEvoOptimizer)
        verifier = IndustrialGraphVerifier(adapter=None, task_type=task)
        optimizer.graph_verifier = verifier
        if planned:
            optimizer.mutation = object.__new__(PlannedMutation)
            optimizer.mutation.raw_input = None
        pipeline = _pipeline(*chain)

        def check(objective):
            assert not verifier.verify(pipeline)
            assert set(verifier.last_report.error_codes) & {
                'input_data_type_mismatch', 'data_type_mismatch', 'feature_type_mismatch'}
            assert all(node.fitted_operation is None for node in pipeline.nodes)
            if planned:
                assert optimizer.mutation.raw_input == RawInputSpec('ts', True)
            return ['checked']

        optimizer._optimise = check
        assert optimizer.optimise(_objective(task)) == ['checked']
        assert verifier.input_data_type is None
        assert verifier.model_feature_flow is ModelFeatureFlow.DECLARED_OUTPUT
        if planned:
            assert optimizer.mutation.raw_input is None


@pytest.mark.parametrize('fail', [False, True])
def test_bound_input_context_restores_explicit_state_even_after_error(fail):
    optimizer = object.__new__(IndustrialEvoOptimizer)
    verifier = IndustrialGraphVerifier(adapter=None, task_type='classification', input_data_type='tabular')
    optimizer.graph_verifier = verifier
    optimizer.mutation = object.__new__(PlannedMutation)
    previous = optimizer.mutation.raw_input = RawInputSpec('image', False)

    def check(objective):
        assert verifier.input_data_type == 'ts'
        assert verifier.model_feature_flow is ModelFeatureFlow.TENSOR_FEATURES
        if fail:
            raise RuntimeError('expected evaluation error')
        return []

    optimizer._optimise = check
    if fail:
        with pytest.raises(RuntimeError, match='expected evaluation error'):
            optimizer.optimise(_objective())
    else:
        optimizer.optimise(_objective())
    assert verifier.input_data_type == 'tabular'
    assert verifier.model_feature_flow is ModelFeatureFlow.DECLARED_OUTPUT
    assert optimizer.mutation.raw_input is previous


@pytest.mark.parametrize('chain', [
    ('pdl_clf',), ('quantile_extractor', 'logit'), ('quantile_extractor', 'pdl_clf', 'logit'),
])
def test_input_admission_preserves_supported_feature_paths(chain):
    with industrial_extension_scope():
        verifier = IndustrialGraphVerifier(adapter=None, task_type='classification', input_data_type='ts',
                                           model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES)
        assert verifier.verify(_pipeline(*chain))


def test_graph_source_aliases_are_normalized_without_task_guessing():
    with industrial_extension_scope():
        verifier = IndustrialGraphVerifier(adapter=None, task_type='classification', input_data_type='table')
        assert verifier.verify(_pipeline('logit'))
        verifier.input_data_type = None
        assert verifier.verify(_pipeline('logit'))


@pytest.mark.parametrize('chain', [
    ('hankelisation', 'ridge'), ('mssa_forecaster',), ('ssa_forecaster',), ('smoothing', 'mssa_forecaster'),
])
def test_confirmed_temporal_input_preserves_supported_forecasters(chain):
    with industrial_extension_scope():
        verifier = IndustrialGraphVerifier(adapter=None, task_type='ts_forecasting', input_data_type='ts',
                                           model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES)
        assert verifier.verify(_pipeline(*chain))


def test_temporal_strategy_is_not_incorrectly_treated_as_feature_identity():
    with industrial_extension_scope():
        verifier = IndustrialGraphVerifier(adapter=None, task_type='ts_forecasting', input_data_type='ts',
                                           model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES)
        assert verifier.verify(_pipeline('ar', 'ridge'))


@pytest.mark.parametrize('source', [None, 'ts'])
def test_declared_container_mismatch_is_retained_after_pdl(source):
    with industrial_extension_scope():
        verifier = IndustrialGraphVerifier(adapter=None, task_type='classification', input_data_type=source,
                                           model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES)
        assert not verifier.verify(_pipeline('pdl_clf', 'riemann_extractor', 'logit'))
        assert 'data_type_mismatch' in verifier.last_report.error_codes


@pytest.mark.parametrize('name,task,source', [
    ('pdl_clf', 'classification', 'ts'), ('pdl_reg', 'regression', 'ts'),
    ('logit', 'classification', 'tabular'), ('ridge', 'regression', 'tabular'),
])
def test_feature_preservation_capability_matches_real_strategy(name, task, source):
    x = np.arange(160, dtype=np.float32).reshape(10, 16)
    if source == 'ts':
        x = x[:, None, :]
    y = np.arange(10) % 2 if task == 'classification' else np.arange(10, dtype=float)
    data = create_data(x, target=y, task=task, data_type=source, use_cache=False)
    params = ({'model': 'dt' if task == 'classification' else 'dtreg', 'max_pairs': 64, 'random_state': 42}
              if name.startswith('pdl_') else None)
    with industrial_extension_scope():
        pipeline = PipelineBuilder().add_node(name, params=params).build()
        capability = graph_spec_from_runtime(pipeline, task_type=task).nodes[0].capabilities
        assert capability.preserves_input_features
        try:
            pipeline.fit(data)
            predict = replace(data, target=None, state='predict')
            before = predict.features.clone()
            output = pipeline.predict(predict, output_mode='labels' if task == 'classification' else 'default')
            assert output.predict is not None
            torch.testing.assert_close(output.features, before)
            torch.testing.assert_close(predict.features, before)
        finally:
            pipeline.unfit()


@pytest.mark.parametrize('has_valid', [False, True])
def test_initial_input_filter_runs_before_evaluator_and_population_extension(has_valid):
    with industrial_extension_scope():
        optimizer = object.__new__(IndustrialEvoOptimizer)
        optimizer.graph_verifier = IndustrialGraphVerifier(
            adapter=None, task_type='classification', input_data_type='ts',
            model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES)
        bad, good = _pipeline('logit'), _pipeline('quantile_extractor', 'logit')
        optimizer.initial_graphs = [bad, good] if has_valid else [bad]
        optimizer.graph_optimizer_params = SimpleNamespace(pop_size=1)
        optimizer.requirements = SimpleNamespace(static_individual_metadata={})
        optimizer.diagnostics_recorder = EvolutionDiagnosticsRecorder()
        optimizer._extend_population = lambda *args: pytest.fail('invalid initial graph was extended')
        optimizer._update_population = lambda **kwargs: None

        def evaluate(population):
            assert [individual.graph for individual in population] == [good]
            return population

        if has_valid:
            assert len(optimizer._initial_population(evaluate)[0]) == 1
        else:
            with pytest.raises(IndustrialPopulationError, match='confirmed input contract') as failure:
                optimizer._initial_population(lambda population: pytest.fail('invalid graph reached evaluation'))
            assert not failure.value.timed_out
        assert optimizer.diagnostics.summary.validation_issue_counts == {'input_data_type_mismatch': 1}


@pytest.mark.parametrize('has_valid', [False, True])
def test_timeout_never_returns_rejected_prevalidated_assumption(has_valid):
    with industrial_extension_scope():
        optimizer = object.__new__(IndustrialEvoOptimizer)
        optimizer.graph_verifier = IndustrialGraphVerifier(adapter=None, task_type='classification')
        bad, good = _pipeline('logit'), _pipeline('quantile_extractor', 'logit')
        optimizer.initial_graphs = [bad, good] if has_valid else [bad]
        optimizer.initial_graphs_prevalidated = True
        optimizer.timer = nullcontext()
        optimizer.requirements = SimpleNamespace(show_progress=False, num_of_generations=1)
        optimizer.eval_dispatcher = SimpleNamespace(dispatch=lambda objective, timer: object())
        optimizer.log = SimpleNamespace(warning=lambda message: None)

        def timed_out(evaluator):
            raise IndustrialPopulationError('expected timeout', timed_out=True)

        optimizer._initial_population = timed_out
        if has_valid:
            assert optimizer.optimise(_objective()) == [good]
        else:
            with pytest.raises(IndustrialPopulationError, match='expected timeout'):
                optimizer.optimise(_objective())
        assert optimizer.graph_verifier.input_data_type is None


def test_diagnostic_export_error_does_not_leak_confirmed_input():
    optimizer = object.__new__(IndustrialEvoOptimizer)
    optimizer.graph_verifier = IndustrialGraphVerifier(adapter=None, task_type='classification')
    optimizer._optimise = lambda objective: []
    optimizer.diagnostics_output_dir = 'not-written'

    def export(path):
        assert optimizer.graph_verifier.input_data_type is None
        raise OSError('expected export error')

    optimizer.export_diagnostics = export
    with pytest.raises(OSError, match='expected export error'):
        optimizer.optimise(_objective())

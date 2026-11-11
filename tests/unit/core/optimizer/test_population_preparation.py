"""Benchmark protocol parameters must be fixed before candidate seeding."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.opt_history_objects.individual import Individual

from benchmark.industrial.experiments.mutation_planning.population_study.preparation import (
    StudyGraphPreparation, configure_pipeline, fitted_model_parameters,
)
from benchmark.industrial.experiments.mutation_planning.population_study.protocol import load_protocol
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.context import current_evaluation_context
from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
from fedot_ind.core.optimizer.randomness import CandidateRandomness
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


def _graphs():
    parameters = load_protocol().to_record()['operation_parameters']
    adapter = PipelineAdapter()
    with industrial_extension_scope():
        a = PipelineBuilder().add_node('quantile_extractor').add_node('pdl_reg').build()
        b = PipelineBuilder().add_node('quantile_extractor', params=dict(reversed(list(
            parameters['quantile_extractor'].items())))).add_node('pdl_reg', params=parameters['pdl_reg']).build()
        return adapter, parameters, adapter.adapt(a), adapter.adapt(b)


def test_implicit_protocol_overrides_and_json_key_order_have_one_identity():
    adapter, parameters, a, b = _graphs()
    before = deepcopy([node.content for node in a.nodes])
    preparation = StudyGraphPreparation.from_parameters(adapter, parameters, 'regression')
    first, second = preparation(a), preparation(b)
    assert first.configuration_json == second.configuration_json
    assert first.configuration_id == second.configuration_id
    assert CandidateRandomness(777, 'evaluation:' + first.configuration_id, 0).seed == (
        CandidateRandomness(777, 'evaluation:' + second.configuration_id, 0).seed)
    assert [node.content for node in a.nodes] == before
    assert preparation(first.graph).configuration_json == first.configuration_json
    parameters['pdl_reg']['random_state'] = 123  # Constructor detached the caller's policy mapping.
    assert preparation(a).configuration_json == first.configuration_json


def _contextual_objective(pipeline):
    context = current_evaluation_context()
    assert context is not None and context.configuration_json is not None
    root = pipeline.root_node
    assert root.parameters['random_state'] == 42
    return SingleObjFitness(float(np.random.random()))


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
def test_dispatcher_prepares_before_rng_and_preserves_per_call_evidence(backend):
    adapter, parameters, a, b = _graphs()
    dispatcher = LocalEvaluationDispatcher(
        adapter, config=EvaluationExecutionConfig(backend), random_seed=777,
        graph_preparation=StudyGraphPreparation.from_parameters(adapter, parameters, 'regression'))
    result = dispatcher.dispatch(_contextual_objective)([Individual(a), Individual(b)])
    assert len(result) == 2
    first, second = [individual.metadata['evaluation_context'] for individual in result]
    assert first['configuration_id'] == second['configuration_id']
    assert first['seed'] == second['seed']
    assert first['evaluation_id'] != second['evaluation_id']
    assert result[0].fitness.value == result[1].fitness.value
    assert current_evaluation_context() is None
    assert all(task.seed == first['seed'] for task in dispatcher.last_plan.tasks)


def test_parameter_changes_and_branch_multiplicity_are_not_erased():
    adapter, parameters, a, _ = _graphs()
    preparation = StudyGraphPreparation.from_parameters(adapter, parameters, 'regression')
    one = preparation(a)
    parameters['pdl_reg']['random_state'] = 123
    two = StudyGraphPreparation.from_parameters(adapter, parameters, 'regression')(a)
    assert one.configuration_id != two.configuration_id
    assert preparation(a).configuration_id != StudyGraphPreparation.from_parameters(
        adapter, load_protocol().to_record()['operation_parameters'], 'classification')(a).configuration_id
    with industrial_extension_scope():
        builder = PipelineBuilder().add_node('quantile_extractor')
        builder.add_node('quantile_extractor', branch_idx=1).join_branches('pdl_reg')
        parallel = preparation(adapter.adapt(builder.build()))
    assert one.configuration_id != parallel.configuration_id
    nodes = json.loads(parallel.configuration_json)['nodes']
    assert len(nodes) == 3 and len(nodes[0]['parents']) == 2


def test_configure_pipeline_is_detached_idempotent_and_never_fits():
    adapter, parameters, a, _ = _graphs()
    with industrial_extension_scope():
        pipeline = adapter.restore(a)
        configure_pipeline(pipeline, parameters)
        original = pipeline.descriptive_id
        configure_pipeline(pipeline, parameters)
        assert pipeline.descriptive_id == original
        assert not pipeline.is_fitted
        assert fitted_model_parameters(pipeline) == []


def test_native_study_records_actual_model_parameters_before_fedot_cleanup():
    import fedot
    from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
    from golem.core.optimisers.objective import Objective
    from benchmark.industrial.experiments.mutation_planning.population_study.worker import StudyEvaluator
    from fedot_ind.core.optimizer.evaluation.archive import CompletedEvaluationArchive

    features = np.arange(30, dtype=float).reshape(10, 3)
    train = fedot.create_data(features, target=np.sin(features[:, 0]), task='regression')
    validation = fedot.create_data(features[:3] + .1, target=np.sin(features[:3, 0]), task='regression')

    def metric(pipeline, reference_data, **kwargs):
        prediction = pipeline.predict(replace(reference_data, target=None, state='predict')).predict
        return float(np.mean((prediction.numpy().reshape(-1) - reference_data.target.numpy().reshape(-1)) ** 2))

    adapter = PipelineAdapter()
    parameters = load_protocol().to_record()['operation_parameters']
    with industrial_extension_scope():
        graph = adapter.adapt(PipelineBuilder().add_node('pdl_reg').build())
    producer = DataSourceSplitter.build_holdout_producer_from_split(train, validation)
    evaluator = StudyEvaluator(Objective({'mse': metric}), producer, parameters=parameters,
                               task='regression', expected_folds=1, result_cache_size=0, eval_n_jobs=1)
    archive = CompletedEvaluationArchive(Objective({'mse': metric}))
    dispatcher = LocalEvaluationDispatcher(
        adapter, config=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL), random_seed=123,
        graph_preparation=StudyGraphPreparation.from_parameters(adapter, parameters, 'regression'),
        completed_evaluation_callback=archive.record_results)
    result, = dispatcher.dispatch(evaluator.evaluate)([Individual(graph)])
    assert result.fitness.valid
    call, = evaluator.calls
    assert call['fitted_models'][0]['parameters']['random_state'] == 42
    assert call['configuration_id'] == result.metadata['evaluation_context']['configuration_id']
    assert call['evaluation_id'] == result.metadata['evaluation_context']['evaluation_id']
    assert call['seed'] == dispatcher.last_plan.tasks[0].seed
    with industrial_extension_scope():
        assert not adapter.restore(result.graph).is_fitted
        assert not adapter.restore(archive.best_individuals[0].graph).is_fitted
    assert len(archive.entries) == 1

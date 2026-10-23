"""One isolated full-population trial, including native FEDOT fold evaluation."""

import argparse
from dataclasses import replace
from datetime import timedelta
import gc
import dask
import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
import traceback

import numpy as np
import torch
import cv2  # noqa: F401
from threadpoolctl import threadpool_info, threadpool_limits
import psutil

from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from golem.core.optimisers.objective import Objective

from .data import load_study_data, loss_value
from .runtime_contract import require_runtime
from .preparation import (
    StudyGraphPreparation, configure_pipeline, fitted_model_parameters, pipeline_configuration,
    selected_evaluation, selection_summary,
)
from fedot_ind.core.optimizer.evaluation.context import current_evaluation_context
from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
from fedot_ind.core.optimizer.randomness import CandidateRandomness


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class StudyEvaluator(PipelineObjectiveEvaluateWithTensorData):
    def __init__(self, *args, parameters, task, trace_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
        self.task = task
        self.calls = []
        self._fitted_models = []
        self.trace_path = trace_path

    def _record_call(self, call):
        self.calls.append(call)
        if self.trace_path is not None:
            with Path(self.trace_path).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(call, allow_nan=False) + '\n')

    def prepare_graph(self, graph, train_data, fold_id=None, n_jobs=-1, *, prediction_cache=None):
        prepared = super().prepare_graph(graph, train_data, fold_id, n_jobs, prediction_cache=prediction_cache)
        self._fitted_models = fitted_model_parameters(prepared)
        return prepared

    def evaluate(self, graph):
        started = perf_counter()
        self._fitted_models = []
        context = current_evaluation_context()
        evidence = {} if context is None else context.to_record()
        try:
            if context is None or context.configuration_json is None:
                raise ValueError('Study evaluation requires explicit prepared configuration evidence')
            snapshot = PreparedEvaluationGraph(graph, canonical_json(pipeline_configuration(graph, self.task)))
            if snapshot.configuration_id != context.configuration_id:
                raise ValueError('Evaluated configuration differs from its RNG identity')
            fitness = super().evaluate(graph)
        except Exception as error:
            self._record_call({**evidence, 'graph_id': graph.descriptive_id, 'valid': False,
                               'seconds': perf_counter() - started, 'outcome': 'RaisedException',
                               'fold_count': 0, 'failure': f'{type(error).__name__}: {error}'})
            raise
        self._record_call({**evidence, 'graph_id': graph.descriptive_id, 'valid': bool(fitness.valid),
                           'fitness': float(fitness.value) if fitness.valid else None,
                           'seconds': perf_counter() - started, 'outcome': type(self.last_outcome).__name__,
                           'fold_count': len(getattr(self.last_outcome, 'folds', ())),
                           'failure': str(getattr(self.last_outcome, 'failure', '')),
                           'fitted_models': self._fitted_models})
        return fitness


class PeakRSS:
    def __init__(self):
        self.stop = Event()
        self.peak = 0
        self.process = psutil.Process()
        self.thread = Thread(target=self._sample, daemon=True)

    def _read(self):
        try:
            total = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.peak = max(total, self.peak)
        except psutil.NoSuchProcess:
            pass

    def _sample(self):
        while not self.stop.wait(0.01):
            self._read()

    def __enter__(self):
        self._read()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self._read()


def run_trial(config, job, root, *, train_only=False, forecast_horizon=None):
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.configuration import (
        EvolutionConfig, ExecutionPolicy, MutationPlanningPolicy, MutationStrategy, ResourceBudget,
    )
    from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
    from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder
    from fedot_ind.core.optimizer.randomness import operator_random_scope
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import run_with_feature_reuse
    from benchmark.industrial.experiments.mutation_planning.runner import _environment

    torch.set_num_threads(1)
    require_runtime(torch)
    torch.manual_seed(job['seed'])
    if train_only:
        from .selection_recheck.data import load_training_data
        data = load_training_data(config, job['scenario'], job['seed'], forecast_horizon=forecast_horizon)
    else:
        data = load_study_data(config, job['scenario'], job['seed'])
    task = job['scenario']['task']
    parameters = config['operation_parameters']

    def metric(pipeline, reference_data, **kwargs):
        output = run_with_feature_reuse(
            pipeline.predict, replace(reference_data, target=None, state='predict'),
            output_mode='labels' if task == 'classification' else 'default')
        return loss_value(task, numpy(reference_data.target), numpy(output.predict), data.scale)

    producer = DataSourceSplitter.build_holdout_producer_from_split(data.train, data.validation)
    objective = Objective({'heldout_loss': metric})
    evaluator = StudyEvaluator(objective, producer, expected_folds=1, result_cache_size=0,
                               eval_n_jobs=1, parameters=parameters, task=task,
                               trace_path=root / 'objective_calls.jsonl')
    scenario = {'name': job['scenario']['name'], 'task': task, 'operations': config['operations'][task],
                'initial_chain': config['initial_chains'][task][0]}
    with (industrial_extension_scope(), threadpool_limits(limits=1), operator_random_scope(job['seed']),
          dask.config.set(scheduler='synchronous')):
        requirements, params, _ = _environment(scenario, config)
        if task == 'ts_forecasting':
            params.advisor.task = data.train.task
        requirements.num_of_generations = config['generations'] - 1
        requirements.timeout = timedelta(seconds=config['search_seconds'])
        requirements.start_depth = requirements.max_depth = config['max_depth']
        requirements.early_stopping_iterations = 100
        requirements.agent_dir = str(root / 'agent')
        graphs = []
        for chain in config['initial_chains'][task]:
            builder = PipelineBuilder()
            for name in chain:
                builder.add_node(name, params=parameters.get(name))
            pipeline = builder.build()
            configure_pipeline(pipeline, parameters)
            # Identical warmup in both fresh processes, excluded from measured search.
            run_with_feature_reuse(pipeline.fit, data.train)
            run_with_feature_reuse(pipeline.predict, replace(data.validation, target=None, state='predict'))
            pipeline.unfit()
            graphs.append(params.adapter.adapt(pipeline))
        recorder = EvolutionDiagnosticsRecorder()
        algorithm = GPAlgorithmParameters(
            pop_size=config['population_size'], max_pop_size=config['population_size'], offspring_rate=1.,
            crossover_prob=0., mutation_prob=1., variable_mutation_num=False, max_num_of_operator_attempts=8,
            mutation_types=[MutationTypesEnum.single_change,
                            MutationTypesEnum.single_add, MutationTypesEnum.single_drop],
        )
        optimizer = IndustrialEvoOptimizer(
            objective, graphs, requirements, params, algorithm,
            EvolutionConfig(random_seed=job['seed'], mutation_strategy=MutationStrategy.GROWTH,
                            mutation_planning=MutationPlanningPolicy(
                                enabled=job['mode'] == 'planned', max_candidates=config['max_candidates'],
                                max_nodes=config['max_nodes']),
                            resource_budget=ResourceBudget(config['population_size'], config['mutation_attempts'], 32),
                            execution_policy=ExecutionPolicy(retry_initial_population_without_timer=False,
                                                             diagnostics_output_dir=str(root / 'diagnostics')),
                            evaluation_execution=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL)),
            diagnostics_recorder=recorder,
            evaluation_graph_preparation=StudyGraphPreparation.from_parameters(params.adapter, parameters, task))
        generations = []
        previous_calls = 0
        previous_events = 0

        def generation_callback(population, owner):
            nonlocal previous_calls, previous_events
            new_calls = evaluator.calls[previous_calls:]
            events = recorder.snapshot().observations
            evaluations = [event for event in events[previous_events:] if event.event_type.value == 'evaluation']
            generations.append({'generation': owner.current_generation_num,
                                'is_final': bool(owner.final_individuals), 'population_size': len(population),
                                'valid_fitness_count': sum(bool(ind.fitness.valid) for ind in population),
                                'new_successes': sum(event.status == 'succeeded' for event in evaluations),
                                'objective_attempts': len(new_calls),
                                'graph_ids': [ind.graph.descriptive_id for ind in population],
                                'members': [{'candidate_id': str(ind.uid), 'fitness': float(ind.fitness.value),
                                             'evaluation_id': ind.metadata['evaluation_context']['evaluation_id']}
                                            for ind in population]})
            previous_calls = len(evaluator.calls)
            previous_events = len(events)

        optimizer.set_iteration_callback(generation_callback)
        with PeakRSS() as memory:
            started = perf_counter()
            print('search-start', flush=True)
            winner = optimizer.optimise(evaluator.evaluate)[0]
            search_seconds = perf_counter() - started
            initial_seconds = optimizer.timer.init_time * 60
            clock_seconds = optimizer.timer.seconds_from_start
            search_clock = {
                'initial_population_seconds': initial_seconds,
                'elapsed_seconds': clock_seconds,
                'evolution_seconds': clock_seconds - initial_seconds,
                'completed_evolved_generations': len(optimizer.evaluated_population) - 1,
                'time_limit_reached': optimizer.timer.process_terminated,
            }
            print('search-finished', search_seconds, flush=True)
            pipeline = configure_pipeline(params.adapter.restore(winner), parameters)
            selected_configuration = PreparedEvaluationGraph(
                winner, canonical_json(pipeline_configuration(pipeline, task)))
            individual = optimizer.best_individuals[0]
            selection = {'candidate_id': str(individual.uid), 'fitness': float(individual.fitness.value),
                         'configuration': json.loads(selected_configuration.configuration_json),
                         'configuration_id': selected_configuration.configuration_id,
                         'evaluation_context': individual.metadata.get('evaluation_context')}
            (root / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False), encoding='utf-8')
            selected_context = selected_evaluation(individual, selected_configuration, evaluator.calls)
            entries = [entry.to_record() for entry in optimizer.completed_evaluations.entries]
            audit = selection_summary(selection, evaluator.calls, generations, completed_entries=entries)
            if not audit['selected_is_observed_minimum']:
                raise ValueError('Selected candidate is not the minimum of completed evaluations')
            selection = {**selection, **audit}
            (root / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False), encoding='utf-8')
            if train_only:
                pipeline.unfit()
                return {**job, 'status': 'succeeded', 'scope': 'TRAIN-only', 'source': data.source,
                        'selected_graph': winner.descriptive_id, 'selected_evaluation': selected_context,
                        'selection': selection,
                        'completed_archive': entries,
                        'generation_best_fitness': float(optimizer.generations.best_individuals[0].fitness.value),
                        'objective_calls': evaluator.calls, 'generations': generations,
                        'search_seconds': search_seconds, 'search_clock': search_clock,
                        'diagnostics': optimizer.diagnostics.summary.to_record(), **require_runtime(torch)}
            pipeline.unfit()
            refit_seed = CandidateRandomness(job['seed'], 'refit:' + selected_configuration.configuration_id, 0).seed
            started = perf_counter()
            with operator_random_scope(refit_seed):
                run_with_feature_reuse(pipeline.fit, data.refit)
            refit_seconds = perf_counter() - started
            refit_models = fitted_model_parameters(pipeline)
            started = perf_counter()
            output = run_with_feature_reuse(
                pipeline.predict, replace(data.test, target=None, state='predict'),
                output_mode='labels' if task == 'classification' else 'default')
            predict_seconds = perf_counter() - started
            predictions = numpy(output.predict)
            quality = loss_value(task, numpy(data.test.target), predictions, data.scale)
            if task == 'classification':
                y = numpy(data.refit.target).astype(int).reshape(-1)
                baseline_prediction = np.full(len(numpy(data.test.target)), np.bincount(y).argmax())
            elif task == 'regression':
                baseline_prediction = np.full(len(numpy(data.test.target)), np.median(numpy(data.refit.target)))
            else:
                baseline_prediction = np.full(len(numpy(data.test.target).reshape(-1)),
                                              numpy(data.refit.features).reshape(-1)[-1])
            baseline = loss_value(task, numpy(data.test.target), baseline_prediction, data.scale)
            pipeline.unfit()
        summary = optimizer.diagnostics.summary.to_record()
        gc.collect()
        runtime = require_runtime(torch)
        return {**job, 'status': 'succeeded', 'quality_loss': quality, 'baseline_loss': baseline,
                'search_seconds': search_seconds, 'search_clock': search_clock, 'refit_seconds': refit_seconds,
                'search_budget_overrun_seconds': max(0., search_seconds - config['search_seconds']),
                'predict_seconds': predict_seconds, 'peak_rss_bytes': memory.peak,
                'memory_scope': 'process_tree_rss_sampled_10ms', 'source': data.source,
                'selected_graph': winner.descriptive_id, 'generations': generations,
                'selected_evaluation': selected_context,
                'selection': selection, 'completed_archive': entries,
                'generation_best_fitness': float(optimizer.generations.best_individuals[0].fitness.value),
                'refit': {'seed': refit_seed, 'configuration_id': selected_configuration.configuration_id,
                          'fitted_models': refit_models},
                'objective_calls': evaluator.calls, 'diagnostics': summary,
                'test_target': numpy(data.test.target).reshape(-1).tolist(),
                'test_prediction': predictions.tolist(), 'threadpools': threadpool_info(), **runtime}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.job.read_text(encoding='utf-8'))
    root = args.job.parent
    accesses = None
    try:
        if payload.get('scope') == 'TRAIN-only':
            from .selection_recheck.data import training_data_guard
            with training_data_guard(payload['config'], payload['job']['scenario']) as accesses:
                result = run_trial(payload['config'], payload['job'], root, train_only=True,
                                   forecast_horizon=payload.get('forecast_horizon'))
        else:
            result = run_trial(payload['config'], payload['job'], root)
    except Exception as error:
        result = {**payload['job'], 'status': 'failed', 'error_type': type(error).__name__,
                  'error_message': str(error), 'traceback': traceback.format_exc()}
    if accesses is not None:
        result['data_access'] = accesses
    temporary = root / 'result.partial'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(root / 'result.json')
    return int(result['status'] != 'succeeded')


if __name__ == '__main__':
    raise SystemExit(main())

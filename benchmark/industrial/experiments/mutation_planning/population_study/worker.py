"""One isolated full-population trial, including native FEDOT fold evaluation."""

import argparse
from copy import deepcopy
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


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def configure_pipeline(pipeline, parameters):
    for node in pipeline.nodes:
        name = node.operation.operation_type
        if name in parameters:
            current = node.parameters if isinstance(node.parameters, dict) else {}
            node.parameters = {**current, **deepcopy(parameters[name])}
    return pipeline


class StudyEvaluator(PipelineObjectiveEvaluateWithTensorData):
    def __init__(self, *args, parameters, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
        self.calls = []

    def evaluate(self, graph):
        started = perf_counter()
        try:
            configure_pipeline(graph, self.parameters)
            fitness = super().evaluate(graph)
        except Exception as error:
            self.calls.append({'graph_id': graph.descriptive_id, 'valid': False,
                               'seconds': perf_counter() - started, 'outcome': 'RaisedException',
                               'fold_count': 0, 'failure': f'{type(error).__name__}: {error}'})
            raise
        self.calls.append({'graph_id': graph.descriptive_id, 'valid': bool(fitness.valid),
                           'fitness': float(fitness.value) if fitness.valid else None,
                           'seconds': perf_counter() - started, 'outcome': type(self.last_outcome).__name__,
                           'fold_count': len(getattr(self.last_outcome, 'folds', ())),
                           'failure': str(getattr(self.last_outcome, 'failure', ''))})
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


def run_trial(config, job, root):
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
    from benchmark.industrial.experiments.mutation_planning.runner import _environment

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('CPU study admission failed: CUDA must be hidden before model training')
    torch.manual_seed(job['seed'])
    data = load_study_data(config, job['scenario'], job['seed'])
    task = job['scenario']['task']
    parameters = config['operation_parameters']
    test_input = replace(data.test, target=None, state='predict')

    def metric(pipeline, reference_data, **kwargs):
        output = pipeline.predict(replace(reference_data, target=None, state='predict'),
                                  output_mode='labels' if task == 'classification' else 'default')
        return loss_value(task, numpy(reference_data.target), numpy(output.predict), data.scale)

    producer = DataSourceSplitter.build_holdout_producer_from_split(data.train, data.validation)
    objective = Objective({'heldout_loss': metric})
    evaluator = StudyEvaluator(objective, producer, expected_folds=1, result_cache_size=0,
                               eval_n_jobs=1, parameters=parameters)
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
            # Identical warmup in both fresh processes, excluded from measured search.
            pipeline.fit(data.train)
            pipeline.predict(replace(data.validation, target=None, state='predict'))
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
            diagnostics_recorder=recorder)
        generations = []
        previous_calls = 0
        previous_events = 0

        def generation_callback(population, owner):
            nonlocal previous_calls, previous_events
            new_calls = evaluator.calls[previous_calls:]
            events = recorder.snapshot().observations
            evaluations = [event for event in events[previous_events:] if event.event_type.value == 'evaluation']
            generations.append({'generation': owner.current_generation_num, 'population_size': len(population),
                                'valid_fitness_count': sum(bool(ind.fitness.valid) for ind in population),
                                'new_successes': sum(event.status == 'succeeded' for event in evaluations),
                                'objective_attempts': len(new_calls),
                                'graph_ids': [ind.graph.descriptive_id for ind in population]})
            previous_calls = len(evaluator.calls)
            previous_events = len(events)

        optimizer.set_iteration_callback(generation_callback)
        with PeakRSS() as memory:
            started = perf_counter()
            print('search-start', flush=True)
            winner = optimizer.optimise(evaluator.evaluate)[0]
            search_seconds = perf_counter() - started
            print('search-finished', search_seconds, flush=True)
            pipeline = configure_pipeline(params.adapter.restore(winner), parameters)
            pipeline.unfit()
            started = perf_counter()
            pipeline.fit(data.refit)
            refit_seconds = perf_counter() - started
            started = perf_counter()
            output = pipeline.predict(test_input, output_mode='labels' if task == 'classification' else 'default')
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
        return {**job, 'status': 'succeeded', 'quality_loss': quality, 'baseline_loss': baseline,
                'search_seconds': search_seconds, 'refit_seconds': refit_seconds,
                'search_budget_overrun_seconds': max(0., search_seconds - config['search_seconds']),
                'predict_seconds': predict_seconds, 'peak_rss_bytes': memory.peak,
                'memory_scope': 'process_tree_rss_sampled_10ms', 'source': data.source,
                'selected_graph': winner.descriptive_id, 'generations': generations,
                'objective_calls': evaluator.calls, 'diagnostics': summary,
                'test_target': numpy(data.test.target).reshape(-1).tolist(),
                'test_prediction': predictions.tolist(), 'threadpools': threadpool_info(),
                'torch_threads': torch.get_num_threads(), 'cuda_visible': torch.cuda.is_available()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.job.read_text(encoding='utf-8'))
    root = args.job.parent
    try:
        result = run_trial(payload['config'], payload['job'], root)
    except Exception as error:
        result = {**payload['job'], 'status': 'failed', 'error_type': type(error).__name__,
                  'error_message': str(error), 'traceback': traceback.format_exc()}
    temporary = root / 'result.partial'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(root / 'result.json')
    return int(result['status'] != 'succeeded')


if __name__ == '__main__':
    raise SystemExit(main())

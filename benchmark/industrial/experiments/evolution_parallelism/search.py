"""Actual Industrial evolution on frozen TRAIN folds, with explicit executors."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import timedelta
from importlib.metadata import version
from hashlib import sha256
import json
import os
import sys
from pathlib import Path
from time import perf_counter, time

import numpy as np

from .config import load_defaults
from .clock import absolute_search_timer, search_complete
from .data import load_data
from .operation_timing import operation_trace


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def loss_value(task, truth, prediction):
    from sklearn.metrics import balanced_accuracy_score, mean_squared_error
    truth, prediction = to_numpy(truth).reshape(-1), to_numpy(prediction).reshape(-1)
    if len(truth) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError('Invalid prediction shape or nonfinite values')
    return (1 - float(balanced_accuracy_score(truth, prediction)) if task == 'classification'
            else float(mean_squared_error(truth, prediction)**.5))


@dataclass
class FoldSource:
    folds: tuple

    def __call__(self):
        return iter(self.folds)


@dataclass
class Metric:
    task: str

    def __call__(self, pipeline, reference_data, **kwargs):
        from dataclasses import replace
        from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import run_with_feature_reuse
        output = run_with_feature_reuse(
            pipeline.predict, replace(reference_data, target=None, state='predict'),
            output_mode='labels' if self.task == 'classification' else 'default')
        return loss_value(self.task, reference_data.target, output.predict)


@dataclass
class MeasuredObjective:
    native: object
    trace_dir: str

    def __call__(self, graph):
        import dask
        from fedot_ind.core.optimizer.evaluation.context import current_evaluation_context
        context = current_evaluation_context()
        identity = context.to_record() if context is not None else {}
        start = perf_counter()
        fitness = None
        try:
            with operation_trace(self.trace_dir, identity), dask.config.set(scheduler='synchronous'):
                fitness = self.native.evaluate(graph)
            return fitness
        finally:
            record = {**identity, 'pid': os.getpid(), 'seconds': perf_counter() - start,
                      'valid': bool(fitness is not None and fitness.valid),
                      'fitness': float(fitness.value) if fitness is not None and fitness.valid else None,
                      'outcome': str(self.native.last_outcome)}
            with (Path(self.trace_dir) / f'evaluations_{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False, default=str) + '\n')


def build_environment(settings, task, workers):
    from fedot.core.pipelines.adapters import PipelineAdapter
    from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
    from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
    from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
    from fedot.core.pipelines.verification import verifier_for_task
    from golem.core.optimisers.optimizer import GraphGenerationParams
    from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier

    operations = settings['tasks'][task.task_type.name]['operations']
    requirements = PipelineComposerRequirements(
        primary=operations, secondary=operations, max_depth=settings['max_depth'],
        max_arity=settings['max_arity'], n_jobs=workers, show_progress=False, keep_history=False,
        early_stopping_timeout=settings['budget_seconds'] / 60.)
    advisor = PipelineChangeAdvisor(task)
    params = GraphGenerationParams(adapter=PipelineAdapter(), advisor=advisor,
                                   node_factory=PipelineOptNodeFactory(requirements, advisor))
    params.verifier = IndustrialGraphVerifier(adapter=params.adapter, task_type=task.task_type,
                                              legacy_verifier=verifier_for_task(task.task_type, params.adapter))
    return requirements, params


def run_search(config, output_dir):
    import torch
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
    from golem.core.optimisers.objective import Objective
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, ExecutionPolicy, MutationPlanningPolicy, ResourceBudget
    from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
    from fedot_ind.core.optimizer.evaluation.dask_config import DaskExecutionConfig
    from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder
    from fedot_ind.core.optimizer.randomness import operator_random_scope
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from benchmark.industrial.experiments.mutation_planning.population_study.preparation import (
        StudyGraphPreparation, configure_pipeline, pipeline_configuration,
    )
    from .monitoring import ProcessTreeMonitor, dask_diagnostics

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    traces = root / 'traces'
    traces.mkdir()
    settings = load_defaults()
    from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest
    from fedot.extensions.parameter_rules import resolve_extension_params
    manifest = build_industrial_extension_manifest()
    declarations = {spec.name: spec for spec in (*manifest.models, *manifest.transforms)}
    for name, parameters in settings['operation_parameters'].items():
        if name in declarations:
            resolved = resolve_extension_params(declarations[name], parameters)
            if not resolved.is_right():
                raise ValueError(f'Invalid experiment parameters for {name}: {resolved}')
    implementation_paths = [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('defaults.json')]
    from fedot_ind.core.optimizer.evaluation import dask_executor, dask_dispatcher
    implementation_paths.extend((Path(dask_executor.__file__), Path(dask_dispatcher.__file__)))
    implementation_paths.extend(Path('fedot_ind/core/optimizer').rglob('*.py'))
    implementation_paths.extend((Path('fedot_ind/core/operation/IndustrialCachableOperation.py'),
                                 Path('fedot_ind/core/repository/data/default_operation_params.json')))
    implementation_paths.append(
        Path('benchmark/industrial/experiments/mutation_planning/population_study/preparation.py'))
    source_hashes = {str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in implementation_paths}
    write_json(root / 'implementation.json', source_hashes)
    write_json(root / 'config.json', {'request': asdict(config), 'settings': settings})
    flags_before = {'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                    'deterministic': torch.are_deterministic_algorithms_enabled()}
    started_all = perf_counter()
    launch_epoch = float(os.environ.get('INDUSTRIAL_SEARCH_START_UNIX', time()))
    deadline_epoch = float(os.environ.get('INDUSTRIAL_SEARCH_DEADLINE_UNIX', launch_epoch + config.budget_seconds))
    runtime_versions = {name: version(name) for name in
                        ('fedot-ind', 'fedot', 'sktime', 'torch', 'dask', 'distributed', 'scikit-learn')}
    write_json(root / 'environment.json', {'python': sys.version, 'executable': sys.executable,
                                           'packages': runtime_versions, 'torch_flags': flags_before,
                                           'native_thread_environment': {name: os.environ.get(name) for name in
                                                                         ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}})
    data = load_data(config.task, config.seed, include_test=False)
    write_json(root / 'source.json', data.source)
    parameters = deepcopy(settings['operation_parameters'])
    if config.task == 'ts_forecasting':
        parameters.setdefault('hankelisation', {})['forecast_length'] = 12
    objective = Objective({'loss': Metric(config.task)})
    native = PipelineObjectiveEvaluateWithTensorData(
        objective, FoldSource(data.folds), expected_folds=len(data.folds), result_cache_size=0, eval_n_jobs=1)
    measured = MeasuredObjective(native, str(traces.resolve()))
    workers = 1 if config.backend == 'sequential' else config.workers
    dask_config = DaskExecutionConfig(retries=1, memory_limit=4 * 1024**3)
    batches, generations, dask_records = [], [], []
    startup_seconds = 0.

    @contextmanager
    def observe(session):
        directory = root / 'dask' / f'batch_{len(batches):03d}'
        batches.append(directory)
        with dask_diagnostics(session.client, directory) as record:
            dask_records.append(record)
            yield record

    with ExitStack() as stack:
        memory = stack.enter_context(ProcessTreeMonitor(root))
        stack.enter_context(industrial_extension_scope())
        stack.enter_context(operator_random_scope(config.seed))
        if config.backend == 'dask':
            from fedot_ind.core.optimizer.evaluation.dask_executor import DaskEvaluationSession
            from fedot_ind.core.optimizer.evaluation.dask_dispatcher import dask_session_scope
            if config.lifecycle == 'search':
                start = perf_counter()
                session = stack.enter_context(DaskEvaluationSession(workers, dask_config))
                startup_seconds = perf_counter() - start
                stack.enter_context(dask_session_scope(session))
                external_client = session.client
                record = stack.enter_context(dask_diagnostics(session.client, root / 'dask' / 'search'))
                dask_records.append(record)
            else:
                stack.enter_context(dask_session_scope(observer=observe))
        requirements, params = build_environment(settings, data.refit.task, workers)
        if time() >= deadline_epoch:
            raise TimeoutError('Search budget consumed during setup')
        requirements.timeout = timedelta(seconds=deadline_epoch - time())
        requirements.num_of_generations = config.generations if config.protocol == 'generations' else 1000
        requirements.early_stopping_iterations = 2000
        requirements.agent_dir = str(root / 'agent')
        initial = []
        for chain in settings['tasks'][config.task]['initial_chains']:
            builder = PipelineBuilder()
            for operation in chain:
                builder.add_node(operation, params=parameters.get(operation))
            initial.append(params.adapter.adapt(builder.build()))

        class JournalRecorder(EvolutionDiagnosticsRecorder):
            def _record(self, event_type, **values):
                event = super()._record(event_type, **values)
                with (root / 'coordinator_events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event.to_record(), allow_nan=False) + '\n')
                return event

        recorder = JournalRecorder()
        algorithm = GPAlgorithmParameters(
            pop_size=settings['population_size'], max_pop_size=settings['population_size'],
            offspring_rate=1., crossover_prob=0., mutation_prob=1., variable_mutation_num=False,
            max_num_of_operator_attempts=12,
            mutation_types=[MutationTypesEnum.single_change, MutationTypesEnum.single_add, MutationTypesEnum.single_drop])
        optimizer = IndustrialEvoOptimizer(
            objective, initial, requirements, params, algorithm,
            EvolutionConfig(random_seed=config.seed,
                            mutation_planning=MutationPlanningPolicy(enabled=True, max_candidates=64, max_nodes=6),
                            resource_budget=ResourceBudget(settings['population_size'], 12, 32),
                            execution_policy=ExecutionPolicy(retry_initial_population_without_timer=False,
                                                             diagnostics_output_dir=str(root / 'diagnostics')),
                            evaluation_execution=EvaluationExecutionConfig(
                                EvaluationBackend(config.backend), candidate_timeout_seconds=settings['candidate_seconds'],
                                total_timeout_seconds=config.budget_seconds,
                                dask=dask_config if config.backend == 'dask' else None)),
            diagnostics_recorder=recorder,
            evaluation_graph_preparation=StudyGraphPreparation.from_parameters(params.adapter, parameters, config.task))

        def on_generation(population, owner):
            row = {'generation': owner.current_generation_num, 'elapsed_seconds': time() - launch_epoch,
                   'final': bool(owner.final_individuals), 'population_size': len(population),
                   'best_loss': min(float(ind.fitness.value) for ind in population if ind.fitness.valid),
                   'completed_evaluations': len(owner.completed_evaluations.entries)}
            row['unique_configurations'] = len({json.loads(entry.context_json)['configuration_id']
                                                for entry in owner.completed_evaluations.entries})
            generations.append(row)
            write_json(root / 'generations.json', generations)
            print(json.dumps(row), flush=True)

        optimizer.set_iteration_callback(on_generation)
        optimizer.timer = absolute_search_timer(launch_epoch, deadline_epoch)
        if time() >= deadline_epoch:
            raise TimeoutError('Search budget consumed before evolution')
        write_json(root / 'limits.json', {
            'search_budget_seconds': config.budget_seconds, 'deadline_unix': deadline_epoch,
            'generation_limit': requirements.num_of_generations,
            'stagnation_iterations': requirements.early_stopping_iterations,
            'stagnation_minutes': requirements.early_stopping_timeout,
            'candidate_seconds': settings['candidate_seconds'], 'complete_process_seconds': 1200,
        })

        def record_completions(results):
            optimizer.completed_evaluations.record_results(results)
            write_json(root / 'completed_evaluations.json',
                       [entry.to_record() for entry in optimizer.completed_evaluations.entries])

        optimizer.eval_dispatcher.completed_evaluation_callback = record_completions
        start = perf_counter()
        selected = optimizer.optimise(measured)[0]
        search_seconds = perf_counter() - start
        search_wall_seconds = time() - launch_epoch
        completed = [entry.to_record() for entry in optimizer.completed_evaluations.entries]
        write_json(root / 'completed_evaluations.json', completed)
        winner = configure_pipeline(params.adapter.restore(selected), parameters)
        selection = pipeline_configuration(winner, config.task)
        write_json(root / 'selected_graph.json', selection)
        evolved = len(optimizer.evaluated_population) - 1
        from .final_evaluation import evaluate_selected
        final = evaluate_selected(config, evolved, winner, data.source, root)
        summary = {'status': 'succeeded', 'request': asdict(config), 'source': data.source,
                   'completed_evolved_generations': evolved, 'search_seconds': search_seconds,
                   'search_wall_seconds': search_wall_seconds,
                   'search_budget_overrun_seconds': max(0., search_wall_seconds - config.budget_seconds),
                   'startup_seconds': startup_seconds, 'refit_seconds': final.refit_seconds,
                   'predict_seconds': final.predict_seconds, 'test_loss': final.test_loss,
                   'validation_loss': min(row['fitness'][0] for row in completed),
                   'completed_evaluations': len(completed),
                   'unique_configurations': len({row['evaluation_context']['configuration_id'] for row in completed}),
                   'diagnostics': optimizer.diagnostics.summary.to_record(), 'dask_batches': len(batches),
                   'parameters': parameters, 'torch_flags_before': flags_before,
                   'versions': runtime_versions}
        if config.backend == 'dask' and config.lifecycle == 'search':
            summary['caller_client_still_running_after_search'] = session.client.status == 'running'
    summary['wall_seconds'] = perf_counter() - started_all
    summary['processes'] = memory.summary
    summary['dask_diagnostics'] = [{'path': str(record.output_dir), **record.summary} for record in dask_records]
    if config.backend == 'dask' and config.lifecycle == 'search':
        summary['session_closed_by_caller_after_use'] = external_client.status == 'closed'
    write_json(root / 'dask_diagnostics.json', summary['dask_diagnostics'])
    summary['minimum_generations_met'] = search_complete(evolved, config.generations)
    if not summary['minimum_generations_met']:
        summary['status'] = 'incomplete'
    summary['torch_flags_after'] = {'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                                    'deterministic': torch.are_deterministic_algorithms_enabled()}
    if summary['torch_flags_after'] != flags_before:
        raise RuntimeError('Global Torch settings changed')
    if any(sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in source_hashes.items()):
        raise RuntimeError('Experiment source changed during the run')
    write_json(root / 'summary.json', summary)
    return summary

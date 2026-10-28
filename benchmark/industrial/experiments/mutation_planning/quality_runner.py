"""Explicit short candidate-search verification, not a full evolution benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import tracemalloc

from benchmark.industrial.experiments.mutation_planning.quality_cost import (
    SCHEMA_VERSION, render_quality_cost_markdown, summarize_quality_cost,
)


@dataclass(frozen=True)
class QualityTrialModel:
    name: str
    parameters_json: str


@dataclass(frozen=True)
class QualityTrialConfig:
    seeds: tuple[int, ...]
    tasks: tuple[str, ...]
    sample_count: int
    series_length: int
    evaluations_per_mode: int
    max_candidates: int
    max_depth: int
    max_nodes: int
    max_arity: int
    generators: tuple[str, ...]
    models: tuple[tuple[str, QualityTrialModel], ...]

    def to_record(self):
        return {
            'schema_version': 'industrial_mutation_quality_trial@1',
            'seeds': list(self.seeds), 'tasks': list(self.tasks), 'sample_count': self.sample_count,
            'series_length': self.series_length, 'evaluations_per_mode': self.evaluations_per_mode,
            'max_candidates': self.max_candidates, 'max_depth': self.max_depth,
            'max_nodes': self.max_nodes, 'max_arity': self.max_arity, 'generators': list(self.generators),
            'models': {task: {'name': spec.name, 'parameters': json.loads(spec.parameters_json)}
                       for task, spec in self.models},
        }


def load_quality_trial_config(path=None) -> QualityTrialConfig:
    path = Path(path) if path is not None else Path(__file__).with_name('quality_defaults.json')
    record = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record, dict) or record.get('schema_version') != 'industrial_mutation_quality_trial@1':
        raise ValueError('Unsupported quality trial schema')
    for field in ('sample_count', 'series_length', 'evaluations_per_mode', 'max_candidates',
                  'max_depth', 'max_nodes', 'max_arity'):
        if type(record.get(field)) is not int or record[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if record['sample_count'] < 20 or record['series_length'] < 8:
        raise ValueError('The trial requires at least 20 samples and 8 time points')
    if (not isinstance(record.get('seeds'), list) or not record['seeds']
            or any(type(seed) is not int or seed < 0 for seed in record['seeds'])
            or len(set(record['seeds'])) != len(record['seeds'])):
        raise ValueError('seeds must be unique nonnegative integers')
    if (not isinstance(record.get('tasks'), list) or not record['tasks']
            or any(task not in ('classification', 'regression') for task in record['tasks'])
            or len(set(record['tasks'])) != len(record['tasks'])):
        raise ValueError('tasks must be unique classification/regression values')
    generators = record.get('generators')
    if (not isinstance(generators, list) or len(generators) < 2
            or any(not isinstance(name, str) or not name.strip() for name in generators)
            or len(set(generators)) != len(generators)):
        raise ValueError('Declare at least two distinct generator names')
    models = record.get('models')
    if not isinstance(models, dict):
        raise ValueError('models must be a task-indexed object')
    for task in record['tasks']:
        model = models.get(task)
        if (not isinstance(model, dict) or not isinstance(model.get('name'), str) or not model['name'].strip()
                or not isinstance(model.get('parameters'), dict)):
            raise ValueError(f'Invalid model specification for {task}')
    expected = set(QualityTrialConfig.__dataclass_fields__) | {'schema_version'}
    if set(record) != expected:
        raise ValueError('Quality trial fields must match the declared schema')
    return QualityTrialConfig(
        seeds=tuple(record['seeds']), tasks=tuple(record['tasks']),
        generators=tuple(record['generators']),
        models=tuple((task, QualityTrialModel(models[task]['name'], json.dumps(
            models[task]['parameters'], sort_keys=True, allow_nan=False))) for task in record['tasks']),
        **{name: record[name] for name in expected - {'schema_version', 'seeds', 'tasks', 'generators', 'models'}},
    )


def _data(task, config, seed):
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from fedot.core.data.tensor_data.tensor_data import TensorData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    rng = np.random.default_rng(seed)
    labels = np.arange(config['sample_count']) % 2
    rng.shuffle(labels)
    amplitudes = .5 + labels + rng.uniform(-.1, .1, len(labels))
    axis = np.linspace(0, 2 * np.pi, config['series_length'])
    features = (amplitudes[:, None] * np.sin(axis)[None, :]
                + rng.normal(0, .02, (len(labels), len(axis))))[:, None, :]
    target = labels if task == 'classification' else amplitudes
    indices = np.arange(len(labels))
    train, heldout = train_test_split(indices, test_size=.4, random_state=seed,
                                      stratify=labels if task == 'classification' else None)
    validation, test = train_test_split(heldout, test_size=.5, random_state=seed,
                                        stratify=labels[heldout] if task == 'classification' else None)

    def data(rows, fitting=False):
        return TensorData(task=Task(TaskTypesEnum[task]), data_type=DataTypesEnum.ts,
                          features=torch.as_tensor(features[rows], dtype=torch.float32), idx=rows,
                          target=torch.as_tensor(target[rows]) if fitting else None,
                          state='fit' if fitting else 'predict')
    digest = sha256(features.tobytes() + target.tobytes()).hexdigest()
    return (data(train, True), data(validation), target[validation], data(test), target[test],
            {'sha256': digest, 'train_indices': train.tolist(),
             'validation_indices': validation.tolist(), 'test_indices': test.tolist()})


def _score(task, target, output):
    import numpy as np
    from sklearn.metrics import f1_score, mean_squared_error

    values = output.predict.detach().cpu().numpy()
    if task == 'classification':
        values = values.argmax(axis=1) if values.ndim == 2 and values.shape[1] > 1 else values.reshape(-1)
        return float(f1_score(target, values, zero_division=0))
    return float(np.sqrt(mean_squared_error(target, values.reshape(-1))))


def _trial(task, seed, mode, config, source, *, measurement_scope='python_tracemalloc'):
    from golem.core.optimisers.adaptive.operator_agent import RandomAgent
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from golem.core.optimisers.genetic.operators.mutation import Mutation
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from fedot_ind.core.optimizer.configuration import MutationPlanningPolicy
    from fedot_ind.core.optimizer.mutation_planning import RawInputSpec
    from fedot_ind.core.optimizer.mutation_runtime import PlannedMutation, install_extension_aware_advisor
    from fedot_ind.core.optimizer.randomness import operator_random_scope
    from benchmark.industrial.experiments.mutation_planning.runner import _environment

    train, validation, validation_target, test, target, data_source = _data(task, config, seed)
    model = config['models'][task]
    scenario = {'name': task, 'task': task, 'operations': config['generators'] + [model['name']],
                'initial_chain': [config['generators'][0], model['name']]}
    requirements, params, graph = _environment(scenario, config)
    install_extension_aware_advisor(params)
    actions = [MutationTypesEnum.single_change, MutationTypesEnum.single_add, MutationTypesEnum.single_drop]
    algorithm = GPAlgorithmParameters(mutation_types=actions, mutation_prob=1., variable_mutation_num=False,
                                      max_num_of_operator_attempts=1,
                                      adaptive_mutation_type=RandomAgent(actions, [1 / 3] * 3))
    operator = Mutation(algorithm, requirements, params)
    if mode == 'planned':
        operator = PlannedMutation(operator, params, MutationPlanningPolicy(
            enabled=True, max_candidates=config['max_candidates'], max_nodes=config['max_nodes']),
            raw_input=RawInputSpec('ts', True))
    errors, observations = [], []
    best, best_quality = None, None
    started = perf_counter()
    tracemalloc.start()
    try:
        for attempt in range(config['evaluations_per_mode']):
            try:
                with operator_random_scope(seed + attempt):
                    candidate = Individual(deepcopy(graph)) if attempt == 0 else operator(Individual(graph))
                    if not isinstance(candidate, Individual) or not params.verifier(candidate.graph):
                        raise ValueError('No admissible changed proposal')
                    if attempt and candidate.graph == graph:
                        raise ValueError('Mutation returned the unchanged parent')
                    pipeline = params.adapter.restore(candidate.graph)
                    pipeline.root_node.parameters = deepcopy(model['parameters'])
                    pipeline.fit(train)
                    quality = _score(task, validation_target, pipeline.predict(validation))
                observations.append({'attempt': attempt, 'graph_id': candidate.graph.descriptive_id,
                                     'validation_quality': quality,
                                     'mutation': candidate.metadata.get('mutation_execution')})
                better = best_quality is None or (
                    (quality > best_quality) if task == 'classification' else quality < best_quality)
                if better:
                    best, best_quality = pipeline, quality
            except (ValueError, TypeError, RuntimeError, AttributeError, IndexError) as error:
                errors.append({'attempt': attempt, 'type': type(error).__name__, 'message': str(error),
                               'mutation': operator.last_execution.to_record()
                               if mode == 'planned' and attempt else None})
        train_seconds = perf_counter() - started
        quality, predict_seconds = None, None
        if best is not None:
            try:
                predicted = perf_counter()
                output = best.predict(test)
                predict_seconds = perf_counter() - predicted
                quality = _score(task, target, output)
            except (ValueError, TypeError, RuntimeError, AttributeError, IndexError) as error:
                errors.append({'stage': 'heldout_prediction', 'type': type(error).__name__, 'message': str(error)})
                best = None
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    budget = {key: value for key, value in config.items() if key != 'schema_version'}
    return {
        'task': task, 'scenario': 'one_step_candidate_search', 'dataset': f'synthetic_{task}',
        'seed': seed, 'mode': mode, 'metric': 'f1' if task == 'classification' else 'rmse',
        'greater_is_better': task == 'classification', 'quality': quality,
        'train_seconds': train_seconds, 'predict_seconds': predict_seconds,
        'peak_memory_bytes': peak, 'memory_scope': measurement_scope,
        'status': 'succeeded' if best is not None else 'failed',
        'budget_id': sha256(json.dumps(budget, sort_keys=True).encode()).hexdigest(),
        'source': {**source, 'data': data_source, 'scope': 'one_step_search_with_train_validation_test',
                   'config': config, 'successful_candidates': observations, 'failed_candidates': errors,
                   'training_time_scope': 'proposal_training_and_validation_selection',
                   'selected_on': 'validation_only'},
    }


def run_quality_trial(output_dir, *, execute=False, config_path=None, compatibility_worktree=None):
    """Preview by default. Real runs require an audited explicit FEDOT copy."""
    config = load_quality_trial_config(config_path).to_record()
    if not execute:
        return {'status': 'planned', 'config': config, 'scope': 'one_step_candidate_search_not_full_evolution'}
    if tracemalloc.is_tracing():
        raise ValueError('The trial must own its tracemalloc session')
    if compatibility_worktree is None:
        raise ValueError('An audited compatibility_worktree is required for execution')
    from fedot_ind.integration.fedot.compat import audit_compatibility_worktree
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    import fedot

    audit = audit_compatibility_worktree(compatibility_worktree)
    if not Path(fedot.__file__).resolve().is_relative_to(Path(audit['worktree']) / 'fedot'):
        raise ValueError('Active FEDOT import does not belong to the audited worktree')
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh output directory; existing evidence is never overwritten')
    root.mkdir(parents=True, exist_ok=True)
    import platform
    import numpy as np
    import sklearn
    import torch
    import fedot_ind

    repository = Path(__file__).resolve().parents[4]
    implementations = [Path(__file__), repository / 'fedot_ind/core/optimizer/mutation_runtime.py',
                       *sorted((repository / 'fedot_ind/core/optimizer/mutation_planning').glob('*.py'))]
    source = {
        'fedot_compatibility': audit,
        'environment': {'python': platform.python_version(), 'industrial': str(fedot_ind.__version__),
                        'numpy': str(np.__version__), 'torch': str(torch.__version__),
                        'sklearn': str(sklearn.__version__)},
        'implementation_sha256': {path.relative_to(repository).as_posix(): sha256(path.read_bytes()).hexdigest()
                                  for path in implementations},
    }
    records = []
    with industrial_extension_scope():
        for task in config['tasks']:
            for seed in config['seeds']:
                for mode in (('legacy', 'planned') if seed % 2 == 0 else ('planned', 'legacy')):
                    records.append(_trial(task, seed, mode, config, source))
                    envelope = {'schema_version': SCHEMA_VERSION, 'records': records}
                    (root / 'records.json').write_text(
                        json.dumps(envelope, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    report = summarize_quality_cost(records)
    (root / 'quality_cost.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (root / 'quality_cost.md').write_text(render_quality_cost_markdown(report), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--compatibility-worktree', type=Path)
    args = parser.parse_args(argv)
    report = run_quality_trial(args.output_dir, execute=args.execute, config_path=args.config,
                               compatibility_worktree=args.compatibility_worktree)
    print(json.dumps(report.get('acceptance', report), indent=2))
    return int(any(record['status'] == 'failed' for record in report.get('records', [])))


if __name__ == '__main__':
    raise SystemExit(main())

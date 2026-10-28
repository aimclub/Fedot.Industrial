"""Bounded source-contract rechecks, feature cost profiling and TRAIN-only diagnosis."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import cProfile
from dataclasses import dataclass
import json
from importlib.metadata import version
from pathlib import Path
import platform
import pstats
from statistics import mean, stdev
import sys
from time import perf_counter

import numpy as np

from .data import file_hash, loss_value
from .forecast_quality import _admit
from .forecast_recheck import _diagnostic_snapshot
from .protocol import canonical, experiment_matrix, parse_protocol
from .runner import execute_worker, native_crash_message, write_json


@dataclass(frozen=True)
class DiagnosticConfig:
    recheck_ordinals: tuple[int, ...]
    profile_datasets: tuple[str, ...]
    profile_repeats: int
    stability_datasets: tuple[str, ...]
    stability_seeds: tuple[int, ...]


def parse_config(raw):
    fields = {'schema_version', 'recheck_ordinals', 'profile_datasets', 'profile_repeats',
              'stability_datasets', 'stability_seeds'}
    if type(raw) is not dict or set(raw) != fields or raw['schema_version'] != 'industrial_supervised_diagnostics@1':
        raise ValueError('Diagnostic fields/schema must match the contract')
    for name in ('recheck_ordinals', 'stability_seeds', 'profile_datasets', 'stability_datasets'):
        values = raw[name]
        numeric = name in ('recheck_ordinals', 'stability_seeds')
        if (type(values) is not list or not values
                or any((type(value) is not int or value < 0) if numeric
                       else (type(value) is not str or not value.strip()) for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'{name} must contain unique valid values')
    if type(raw['profile_repeats']) is not int or not 1 <= raw['profile_repeats'] <= 5:
        raise ValueError('profile_repeats must be an integer from 1 to 5')
    if len(raw['stability_seeds']) < 3:
        raise ValueError('Require at least three TRAIN-only splits')
    return DiagnosticConfig(**{name: raw[name] if name == 'profile_repeats' else tuple(raw[name])
                               for name in fields - {'schema_version'}})


def summarize_stability(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['dataset'], row['candidate']].append(row['validation_loss'])
    return [{'dataset': dataset, 'candidate': candidate, 'splits': len(values),
             'mean_loss': mean(values), 'sd_loss': stdev(values) if len(values) > 1 else 0.}
            for (dataset, candidate), values in sorted(groups.items())]


def summarize_selection(records):
    """Audit recorded regression choices, not unobserved alternatives or fitted states.

    A descriptive graph ID may alias multiple evaluations. Report the entire score
    range instead of claiming to identify which fitted episode became the winner.
    TEST errors are descriptive evidence only; they never enter model selection.
    """
    result = []
    for record in records:
        if record['status'] != 'succeeded' or record['scenario']['task'] != 'regression':
            raise ValueError('Selection audit requires successful regression records')
        groups = defaultdict(list)
        for call in record['objective_calls']:
            if not call['valid']:
                continue
            score = call['fitness']
            if type(score) not in (float, int) or not np.isfinite(score) or score < 0:
                raise ValueError('Validation losses must be finite nonnegative numbers')
            groups[call['graph_id']].append(float(score))
        selected = record['selected_graph']
        if not groups or selected not in groups:
            raise ValueError('Selected graph must have an observed valid evaluation')
        best = min(min(values) for values in groups.values())
        selected_best = min(groups[selected])
        alternative = min((min(values) for name, values in groups.items() if name != selected), default=None)
        target = np.asarray(record['test_target'], dtype=float).reshape(-1)
        prediction = np.asarray(record['test_prediction'], dtype=float).reshape(-1)
        scale = record['source']['normalization_scale']
        if (not len(target) or target.shape != prediction.shape or not np.isfinite(target).all()
                or not np.isfinite(prediction).all() or type(scale) not in (int, float)
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError('Require finite equal-length TEST values and a positive TRAIN scale')
        try:
            with np.errstate(over='raise', invalid='raise'):
                squared = np.square((target - prediction) / scale)
                total = float(squared.sum())
                replayed_loss = float(np.sqrt(squared.mean()))
        except FloatingPointError as error:
            raise ValueError('TEST error aggregation overflowed') from error
        order = np.argsort(-squared, kind='stable')
        result.append({
            'dataset': record['scenario']['name'], 'seed': record['seed'], 'mode': record['mode'],
            'ordinal': record['ordinal'], 'selected_graph': selected,
            'observed_validation_minimum': best, 'selected_graph_best_validation_loss': selected_best,
            'selected_graph_contains_observed_minimum': selected_best == best,
            'selected_graph_validation_losses': groups[selected],
            'best_other_graph_validation_loss': alternative,
            'validation_margin_to_other_graph': None if alternative is None else alternative - selected_best,
            'objective_attempts': len(record['objective_calls']), 'distinct_recorded_graphs': len(groups),
            'repeated_graph_scores': [
                {'graph_id': name, 'evaluations': len(values), 'losses': values,
                 'minimum': min(values), 'maximum': max(values), 'range': max(values) - min(values)}
                for name, values in sorted(groups.items()) if len(values) > 1],
            'test_loss_recorded': record['quality_loss'], 'test_loss_replayed': replayed_loss,
            'baseline_loss': record['baseline_loss'], 'test_count': len(target),
            'test_top1_squared_error_fraction': float(squared[order[:1]].sum() / total) if total else 0.,
            'test_top5_squared_error_fraction': float(squared[order[:5]].sum() / total) if total else 0.,
            'test_largest_error_indices': order[:5].tolist(),
            'test_prediction_range': [float(prediction.min()), float(prediction.max())],
            'test_target_range': [float(target.min()), float(target.max())],
        })
    return result


def _train_arrays(study, scenario):
    from benchmark.industrial.datasets.local_io import (
        _load_tabular_split, _load_ts_split, resolve_local_split_paths,
    )
    root = study['classification_root'] if scenario['task'] == 'classification' else study['supervised_root']
    path, _ = resolve_local_split_paths(scenario['name'], data_root=root)
    # Reuse the existing parsers but open only TRAIN, never the resolved TEST path.
    if path.suffix.lower() == '.ts':
        features, target, _ = _load_ts_split(path)
    else:
        features, target = _load_tabular_split(path, separator='\t' if path.suffix.lower() == '.tsv' else ',')
    if scenario['task'] == 'classification':
        from sklearn.preprocessing import LabelEncoder
        target = LabelEncoder().fit_transform(target)
    features, target = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError('Diagnostics require finite original TRAIN values')
    return features, target, path


def _tensor(features, target, task):
    import torch
    from fedot.core.data.tensor_data import TensorData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    values = features[:, None, :] if features.ndim == 2 else features
    return TensorData(features=torch.as_tensor(values, dtype=torch.float32).clone(),
                      target=None if target is None else torch.as_tensor(target).clone(), idx=np.arange(len(values)),
                      data_type=DataTypesEnum.ts, task=Task(TaskTypesEnum[task]),
                      state='predict' if target is None else 'fit')


def _pipeline(kind, task, parameters):
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    builder = PipelineBuilder()
    if kind == 'pdl':
        name = 'pdl_clf' if task == 'classification' else 'pdl_reg'
        builder.add_node(name, params=parameters[name])
    else:
        head = 'logit' if task == 'classification' else 'ridge'
        builder.add_node('quantile_extractor', params=parameters['quantile_extractor'])
        if kind == 'composite':
            builder.add_node('quantile_extractor', branch_idx=1, params=parameters['quantile_extractor'])
            builder.add_node('quantile_extractor_torch', branch_idx=1,
                             params={**parameters['quantile_extractor_torch'], 'task_type': task})
            builder.join_branches(head, params=parameters.get(head))
        elif kind == 'simple':
            builder.add_node(head, params=parameters.get(head))
        else:
            raise ValueError('Unknown diagnostic candidate')
    pipeline = builder.build()
    from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier, ModelFeatureFlow
    verifier = IndustrialGraphVerifier(adapter=None, task_type=task, input_data_type='ts',
                                       model_feature_flow=ModelFeatureFlow.TENSOR_FEATURES)
    if not verifier.verify(pipeline):
        raise ValueError(f'Diagnostic graph rejected: {verifier.last_report.error_codes}')
    return pipeline


@contextmanager
def _runtime():
    import dask
    import torch
    from threadpoolctl import threadpool_limits
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    torch.set_num_threads(1)
    with industrial_extension_scope(), threadpool_limits(limits=1), dask.config.set(scheduler='synchronous'):
        yield


def profile_features(study, scenario, repeats):
    features, target, path = _train_arrays(study, scenario)
    task = scenario['task']
    train, predict = _tensor(features, target, task), _tensor(features, None, task)
    rows = []
    with _runtime():
        for kind in ('simple', 'composite'):
            pipeline = _pipeline(kind, task, study['operation_parameters'])
            try:
                pipeline.fit(train)
                pipeline.predict(predict)
                pipeline.unfit()
                started = perf_counter()
                pipeline.fit(train)
                fitting = perf_counter() - started
                timings = []
                for _ in range(repeats):
                    started = perf_counter()
                    pipeline.predict(predict)
                    timings.append(perf_counter() - started)
                profiler = cProfile.Profile()
                profiler.runcall(pipeline.predict, predict)
                hot = sorted(pstats.Stats(profiler).stats.items(), key=lambda item: item[1][3], reverse=True)[:25]
                rows.append({'dataset': scenario['name'], 'task': task, 'candidate': kind,
                             'source_train': str(path), 'source_sha256': file_hash(path), 'samples': len(target),
                             'graph_id': pipeline.descriptive_id, 'fit_seconds': fitting, 'predict_seconds': timings,
                             'profile_excluded_from_timing': True,
                             'profile': [{'file': key[0], 'line': key[1], 'function': key[2],
                                          'calls': value[1], 'self_seconds': value[2], 'cumulative_seconds': value[3]}
                                         for key, value in hot]})
            finally:
                pipeline.unfit()
    return rows


def diagnose_stability(study, scenario, seeds):
    from sklearn.model_selection import train_test_split
    if scenario['task'] != 'regression':
        raise ValueError('Selection diagnosis is restricted to the approved regression cases')
    features, target, path = _train_arrays(study, scenario)
    rows = []
    with _runtime():
        for seed in seeds:
            training, validation = train_test_split(np.arange(len(target)), test_size=.25, random_state=seed)
            scale = float(np.std(target[training]))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Require positive TRAIN-only scale')
            data = _tensor(features[training], target[training], 'regression')
            predict = _tensor(features[validation], None, 'regression')
            for kind in ('simple', 'composite', 'pdl', 'median'):
                if kind == 'median':
                    prediction = np.full(len(validation), np.median(target[training]))
                else:
                    pipeline = _pipeline(kind, 'regression', study['operation_parameters'])
                    try:
                        pipeline.fit(data)
                        prediction = pipeline.predict(predict).predict.detach().cpu().numpy()
                    finally:
                        pipeline.unfit()
                rows.append({'dataset': scenario['name'], 'candidate': kind, 'seed': seed,
                             'source_train': str(path), 'source_sha256': file_hash(path),
                             'training_indices': training.tolist(), 'validation_indices': validation.tolist(),
                             'normalization_scale': scale, 'validation_target': target[validation].tolist(),
                             'validation_prediction': np.asarray(prediction).reshape(-1).tolist(),
                             'validation_loss': loss_value('regression', target[validation], prediction, scale)})
    return rows


def run_diagnostics(source, output, *, worktree=None, execute=False, config_path=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    config_path = (Path(config_path) if config_path else Path(__file__).with_name(
        'supervised_diagnostics_defaults.json')).resolve()
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    config = parse_config(raw)
    original = json.loads((source / 'frozen_manifest.json').read_text(encoding='utf-8'))
    protocol = parse_protocol(original['protocol'])
    if (original['protocol_sha256'] != protocol.digest
            or canonical(experiment_matrix(protocol)) != canonical(original['matrix'])):
        raise ValueError('Original frozen protocol/matrix mismatch')
    study = protocol.to_record()
    scenarios = {row['name']: row for row in study['scenarios'] if row['task'] != 'ts_forecasting'}
    if not set(config.profile_datasets + config.stability_datasets) <= scenarios.keys():
        raise ValueError('Select only existing supervised scenarios')
    jobs = tuple(row for row in original['matrix'] if row['ordinal'] in config.recheck_ordinals)
    if len(jobs) != len(config.recheck_ordinals) or any(row['scenario']['task'] == 'ts_forecasting' for row in jobs):
        raise ValueError('Select only existing supervised trials')
    plan = {'purpose': 'diagnostic_only_not_full_acceptance', 'config': raw, 'jobs': jobs,
            'original_protocol_sha256': protocol.digest, 'test_used_only_in_frozen_rechecks': True}
    if not execute:
        return plan
    if worktree is None or output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Require an audited worktree and separate diagnostic output')
    audit = _admit(worktree)
    signature = _diagnostic_snapshot()
    defaults_hash = file_hash(config_path)
    source_hash = file_hash(source / 'frozen_manifest.json')
    data_hashes = {path: digest for item in original['data_sources'].values()
                   for path, digest in item['files'].items()}

    def environment():
        return {'python_executable': sys.executable, 'python': platform.python_version(),
                'platform': platform.platform(),
                'packages': {name: version(name) for name in original['environment']['packages']}}

    admitted_environment = environment()

    def check_sources():
        if (signature != _diagnostic_snapshot() or _admit(worktree) != audit
                or environment() != admitted_environment
                or file_hash(config_path) != defaults_hash or file_hash(source / 'frozen_manifest.json') != source_hash
                or any(file_hash(path) != digest for path, digest in data_hashes.items())):
            raise ValueError('Diagnostic implementation, environment, configuration or inputs changed')

    check_sources()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', {
        **plan, 'implementation_sha256': signature, 'compatibility': audit, 'study': study,
        'diagnostic_defaults_sha256': defaults_hash, 'original_manifest_sha256': source_hash,
        'environment': admitted_environment,
    })
    rechecks = []
    for job in jobs:
        check_sources()
        directory = output / f"run_{job['ordinal']:03}"
        directory.mkdir()
        write_json(directory / 'job.json', {'job': job, 'config': study})
        print(f"RECHECK {job['ordinal']} {job['scenario']['name']} {job['mode']}", flush=True)
        code = execute_worker(directory / 'job.json', study['run_seconds'])
        write_json(directory / 'process.json', {'exit_code': code})
        if code != 0 or native_crash_message(directory / 'run.log'):
            raise RuntimeError(f"Recheck {job['ordinal']} did not complete cleanly")
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        if result['source'] != original['data_sources'][f"{job['scenario']['name']}:{job['seed']}"]:
            raise ValueError('Recheck data differs from frozen source')
        rechecks.append(result)
        write_json(output / 'rechecks.json', rechecks)
    profiles, stability = [], []
    for name in config.profile_datasets:
        check_sources()
        print(f'PROFILE TRAIN {name}', flush=True)
        profiles.extend(profile_features(study, scenarios[name], config.profile_repeats))
        write_json(output / 'profiles.json', profiles)
    for name in config.stability_datasets:
        check_sources()
        print(f'STABILITY TRAIN {name}', flush=True)
        stability.extend(diagnose_stability(study, scenarios[name], config.stability_seeds))
        write_json(output / 'stability.json', stability)
    check_sources()
    for row in profiles + stability:
        if file_hash(row['source_train']) != row['source_sha256']:
            raise ValueError('TRAIN source changed during diagnostics')
    summary = {'purpose': plan['purpose'], 'rechecks': len(rechecks),
               'recheck_evaluation_failures': [row['diagnostics']['evaluations_failed'] for row in rechecks],
               'stability': summarize_stability(stability)}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compatibility-worktree', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = run_diagnostics(args.source, args.output, worktree=args.compatibility_worktree,
                             config_path=args.config, execute=args.execute)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Paired CPU measurements of exact statistical reuse, not EVO acceptance."""

import argparse
from dataclasses import asdict
import gc
import json
from importlib.metadata import version
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

import numpy as np

from .data import file_hash
from .forecast_quality import _admit
from .forecast_recheck import _diagnostic_snapshot
from .protocol import parse_protocol
from .runner import write_json
from .supervised_diagnostics import _pipeline, _runtime, _tensor, _train_arrays


def parse_config(raw):
    fields = {'schema_version', 'datasets', 'candidates', 'paired_repeats', 'predict_repeats',
              'seed', 'max_bytes', 'max_entries'}
    if (type(raw) is not dict or set(raw) != fields
            or raw['schema_version'] != 'industrial_feature_reuse_diagnostics@1'):
        raise ValueError('Invalid feature reuse diagnostic schema')
    for name in ('datasets', 'candidates'):
        if (type(raw[name]) is not list or not raw[name] or len(raw[name]) != len(set(raw[name]))
                or any(type(value) is not str or not value.strip() for value in raw[name])):
            raise ValueError(f'{name} must contain unique names')
    if set(raw['candidates']) != {'simple', 'composite'}:
        raise ValueError('Require both simple control and composite candidates')
    for name, low, high in (('paired_repeats', 3, 5), ('predict_repeats', 3, 5), ('seed', 0, 2**32 - 1),
                            ('max_bytes', 1, 128 * 1024 * 1024), ('max_entries', 1, 128)):
        if type(raw[name]) is not int or not low <= raw[name] <= high:
            raise ValueError(f'{name} outside diagnostic bounds')
    return raw


def _snapshot():
    paths = _diagnostic_snapshot()
    path = 'fedot_ind/core/repository/IndustrialDispatcher.py'
    paths[path] = file_hash(path)
    path = Path(__file__).with_name('feature_reuse_defaults.json')
    paths[path.relative_to(Path.cwd()).as_posix() if path.is_relative_to(Path.cwd()) else str(path)] = file_hash(path)
    return paths


def _measure(study, scenario, config, kind, repeat, enabled, train, predict, output):
    import torch
    from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import feature_reuse_scope
    from .worker import PeakRSS

    budget = config['max_bytes'] if enabled else 0
    pipeline = _pipeline(kind, scenario['task'], study['operation_parameters'])
    graph_id = pipeline.descriptive_id
    arrays, timings, statistics = {}, [], []
    prefix = f"{scenario['name']}_{kind}_{repeat}_{'reuse' if enabled else 'reference'}"
    try:
        torch.manual_seed(config['seed'])
        with PeakRSS() as memory:
            started = perf_counter()
            with feature_reuse_scope(max_bytes=budget, max_entries=config['max_entries']) as session:
                result = pipeline.fit(train)
                stats = asdict(session.statistics)
            fitting = perf_counter() - started
            statistics.append(stats)
            arrays['fit_features'] = result.features.detach().cpu().numpy().copy()
            arrays['fit_prediction'] = result.predict.detach().cpu().numpy().copy()
            for index in range(config['predict_repeats']):
                started = perf_counter()
                with feature_reuse_scope(max_bytes=budget, max_entries=config['max_entries']) as session:
                    result = pipeline.predict(predict)
                    stats = asdict(session.statistics)
                timings.append(perf_counter() - started)
                statistics.append(stats)
                arrays[f'predict_features_{index}'] = result.features.detach().cpu().numpy().copy()
                arrays[f'prediction_{index}'] = result.predict.detach().cpu().numpy().copy()
        path = output / f'{prefix}.npz'
        np.savez_compressed(path, **arrays)
        row = {'dataset': scenario['name'], 'task': scenario['task'], 'candidate': kind, 'repeat': repeat,
               'enabled': enabled, 'graph_id': graph_id, 'fit_seconds': fitting, 'predict_seconds': timings,
               'peak_process_rss_bytes': memory.peak, 'cache_statistics': statistics,
               'arrays': path.name, 'arrays_sha256': file_hash(path)}
        return row, arrays
    finally:
        pipeline.unfit()
        gc.collect()


def compare_arrays(reference, reused):
    if reference.keys() != reused.keys():
        raise ValueError('Comparison array fields differ')
    for name in reference:
        a, b = reference[name], reused[name]
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f'Exact output comparison failed: {name}')


def run(source, output, *, worktree=None, execute=False, config_path=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    defaults = Path(config_path) if config_path else Path(__file__).with_name('feature_reuse_defaults.json')
    config = parse_config(json.loads(defaults.read_text(encoding='utf-8')))
    plan = {'purpose': 'diagnostic_only_not_full_acceptance', 'config': config,
            'deterministic_torch_algorithms': True, 'input': 'original_TRAIN_only',
            'scope': 'fresh_storage_for_each_fit_and_predict'}
    if not execute:
        return plan
    if worktree is None or source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Require audited FEDOT and separate new output root')
    import torch
    audit = _admit(worktree)
    original_path = source / 'frozen_manifest.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    protocol = parse_protocol(original['protocol'])
    if protocol.digest != original['protocol_sha256']:
        raise ValueError('Frozen protocol digest differs')
    study = protocol.to_record()
    scenarios = {s['name']: s for s in study['scenarios'] if s['task'] != 'ts_forecasting'}
    if not set(config['datasets']) <= scenarios.keys():
        raise ValueError('Diagnostics must use existing supervised scenarios')
    implementation = _snapshot()
    inputs = {name: _train_arrays(study, scenarios[name]) for name in config['datasets']}
    data_hashes = {str(row[2]): file_hash(row[2]) for row in inputs.values()}
    source_hash, config_hash = file_hash(original_path), file_hash(defaults)
    environment = {name: version(name) for name in original['environment']['packages']}
    if environment != original['environment']['packages']:
        raise ValueError('Package versions differ from the admitted CPU environment')

    def check_sources():
        if (_snapshot() != implementation or _admit(worktree) != audit or file_hash(original_path) != source_hash
                or file_hash(defaults) != config_hash
                or any(file_hash(path) != digest for path, digest in data_hashes.items())):
            raise ValueError('Diagnostic source/configuration/data changed')

    check_sources()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', {
        **plan, 'implementation_sha256': implementation,
        'fedot_compatibility': audit, 'data_sha256': data_hashes, 'study': study,
        'original_manifest_sha256': source_hash, 'defaults_sha256': config_hash,
        'python_executable': sys.executable, 'python': platform.python_version(),
        'environment': environment, 'platform': platform.platform(), 'torch_version': torch.__version__,
    })
    enabled, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    records, pairs = [], []
    try:
        with _runtime(), torch.random.fork_rng(devices=[]):
            torch.use_deterministic_algorithms(True)
            for name in config['datasets']:
                features, target, _ = inputs[name]
                train = _tensor(features, target, scenarios[name]['task'])
                predict = _tensor(features, None, scenarios[name]['task'])
                source_features, source_target = train.features.clone(), train.target.clone()
                for kind in config['candidates']:
                    # Identical warmup, excluded from measured pairs.
                    warmup = _pipeline(kind, scenarios[name]['task'], study['operation_parameters'])
                    try:
                        warmup.fit(train)
                        warmup.predict(predict)
                    finally:
                        warmup.unfit()
                    for repeat in range(config['paired_repeats']):
                        check_sources()
                        print(f'PAIR {name} {kind} repeat={repeat}', flush=True)
                        measured = {}
                        for mode in ((False, True) if repeat % 2 == 0 else (True, False)):
                            measured[mode] = _measure(study, scenarios[name], config, kind, repeat,
                                                      mode, train, predict, output)
                        a, b = measured[False][0], measured[True][0]
                        compare_arrays(measured[False][1], measured[True][1])
                        if a['graph_id'] != b['graph_id']:
                            raise ValueError('Compared graphs differ')
                        if (not torch.equal(train.features, source_features)
                                or not torch.equal(train.target, source_target)):
                            raise ValueError('Caller-owned input mutated')
                        records.extend((a, b))
                        pairs.append({'dataset': name, 'candidate': kind, 'repeat': repeat,
                                      'exact_arrays': True, 'graph_id': a['graph_id'],
                                      'fit_ratio': b['fit_seconds'] / a['fit_seconds'],
                                      'predict_ratio': median(b['predict_seconds']) / median(a['predict_seconds'])})
                        write_json(output / 'records.json', records)
                        write_json(output / 'pairs.json', pairs)
            check_sources()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
    summary = [{'dataset': name, 'candidate': kind,
                'fit_ratio_median': median(p['fit_ratio'] for p in pairs
                                           if p['dataset'] == name and p['candidate'] == kind),
                'predict_ratio_median': median(p['predict_ratio'] for p in pairs
                                               if p['dataset'] == name and p['candidate'] == kind)}
               for name in config['datasets'] for kind in config['candidates']]
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
    print(json.dumps(run(args.source, args.output, worktree=args.compatibility_worktree,
                         execute=args.execute, config_path=args.config), indent=2))


if __name__ == '__main__':
    main()

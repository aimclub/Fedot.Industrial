"""Execute a bounded frozen TRAIN-only selection recheck in isolated CPU processes."""

import argparse
from importlib import resources
import json
from pathlib import Path
import sys

from .data import load_training_data, training_data_guard
from ..data import file_hash
from ..protocol import canonical, experiment_matrix, load_protocol
from ..preparation import completed_objective_calls
from ..runner import (_fedot_source_snapshot, _source_snapshot, _verify_frozen_execution,
                      execute_worker, native_crash_message, write_json)
from ..runtime_contract import require_runtime


def load_settings(path=None):
    source = resources.files(__package__).joinpath('defaults.json') if path is None else Path(path)
    raw = json.loads(source.read_text(encoding='utf-8'))
    schema = raw.get('schema_version')
    datasets = {'industrial_selection_recheck@1': ('D1', 'BitcoinSentiment'),
                'industrial_completed_archive_recheck@1': ('BitcoinSentiment',)}
    if (set(raw) != {'schema_version', 'forecast_horizon', 'horizon_source', 'jobs'}
            or schema not in datasets
            or type(raw['forecast_horizon']) is not int or raw['forecast_horizon'] < 1
            or len(raw['jobs']) != 2 * len(datasets[schema])):
        raise ValueError('Require the approved TRAIN-only job set and an explicit frozen horizon')
    expected = {(name, 42, mode) for name in datasets[schema] for mode in ('legacy', 'planned')}
    if {(job['dataset'], job['seed'], job['mode']) for job in raw['jobs']} != expected:
        raise ValueError('Diagnostic job set differs from the approved four searches')
    return raw


def run(output_dir, config_path, worktree, *, settings_path=None):
    import fedot
    import torch
    from fedot_ind.integration.fedot.compat import audit_compatibility_worktree

    torch.set_num_threads(1)
    runtime = require_runtime(torch)
    settings = load_settings(settings_path)
    protocol = load_protocol(config_path)
    config = protocol.to_record()
    audit = audit_compatibility_worktree(worktree)
    if not Path(fedot.__file__).resolve().is_relative_to(Path(audit['worktree']) / 'fedot'):
        raise ValueError('Active FEDOT must belong to the audited copy')
    matrix = experiment_matrix(protocol)
    jobs = [next(row for row in matrix if row['scenario']['name'] == job['dataset']
                 and row['seed'] == job['seed'] and row['mode'] == job['mode']) for job in settings['jobs']]
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    envelope = {'scope': 'TRAIN-only', 'settings': settings, 'protocol': config,
                'protocol_sha256': protocol.digest, 'matrix': jobs, 'runtime': runtime,
                'python_executable': sys.executable, 'implementation_sha256': _source_snapshot(),
                'fedot_sources_sha256': _fedot_source_snapshot(audit['worktree']), 'fedot_compatibility': audit}
    lock = root / 'code_lock.json'
    if lock.exists():
        if canonical(json.loads(lock.read_text(encoding='utf-8'))) != canonical(envelope):
            raise ValueError('Frozen diagnostic code/configuration changed; use a fresh output root')
    else:
        write_json(lock, envelope)
    manifest = {**envelope, 'code_lock_sha256': file_hash(lock)}
    _verify_frozen_execution(manifest, root)
    sources = {}
    for job in jobs:
        with training_data_guard(config, job['scenario']) as accesses:
            data = load_training_data(config, job['scenario'], job['seed'],
                                      forecast_horizon=settings['forecast_horizon'])
        if accesses['blocked']:
            raise ValueError('Forbidden data access during preflight')
        sources[job['scenario']['name']] = data.source
    manifest['data_sources'] = sources
    frozen = root / 'frozen_manifest.json'
    if frozen.exists() and canonical(json.loads(frozen.read_text(encoding='utf-8'))) != canonical(manifest):
        raise ValueError('Frozen training inputs changed; use a fresh output root')
    write_json(frozen, manifest)
    records = []
    for job in jobs:
        _verify_frozen_execution(manifest, root)
        if require_runtime(torch) != runtime:
            raise ValueError('CPU/Torch runtime changed')
        directory = root / f"{job['scenario']['name']}_{job['seed']}_{job['mode']}"
        directory.mkdir(exist_ok=True)
        path = directory / 'result.json'
        process = directory / 'process.json'
        attempt = directory / 'attempt.json'
        if not path.exists() and not process.exists() and not attempt.exists():
            payload = {'config': config, 'job': job, 'scope': 'TRAIN-only',
                       'forecast_horizon': settings['forecast_horizon']}
            write_json(directory / 'job.json', payload)
            write_json(attempt, {'job': job, 'automatic_retry': False})
            print(f"START {job['scenario']['name']} {job['mode']} TRAIN-only", flush=True)
            try:
                code = execute_worker(directory / 'job.json', config['run_seconds'])
                write_json(process, {'exit_code': code})
            except (RuntimeError, TimeoutError, MemoryError) as error:
                write_json(process, {'exit_code': None, 'error_type': type(error).__name__,
                                     'error_message': str(error)})
        _verify_frozen_execution(manifest, root)
        if not process.exists():
            raise ValueError('Incomplete worker evidence; automatic reruns are forbidden')
        execution = json.loads(process.read_text(encoding='utf-8'))
        code = execution['exit_code']
        record = (json.loads(path.read_text(encoding='utf-8')) if path.exists() else
                  {**job, 'status': 'failed', 'scope': 'TRAIN-only', 'process': execution})
        if record['status'] == 'succeeded':
            if (code != 0 or native_crash_message(directory / 'run.log')
                    or record['source'] != sources[job['scenario']['name']]
                    or record.get('scope') != 'TRAIN-only' or record['data_access']['blocked']
                    or not record['data_access']['opened_training_files']):
                raise ValueError('Worker provenance/data-access contract failed')
            if (not record['selection']['selected_is_observed_minimum']
                    or len(completed_objective_calls(record['objective_calls'], record['completed_archive'])) !=
                    record['diagnostics']['evaluations_succeeded']):
                raise ValueError('Completed archive does not cover all successful evaluations or their minimum')
        records.append(record)
        write_json(root / 'records.json', records)
        print(f"DONE {job['scenario']['name']} {job['mode']} {record['status']} "
              f"selection={record.get('selection', {}).get('fitness')}", flush=True)
    _verify_frozen_execution(manifest, root)
    report = {'scope': 'TRAIN-only', 'status': 'completed' if all(row['status'] == 'succeeded' for row in records)
              else 'incomplete', 'runs': len(records), 'records': records,
              'acceptance_scope': 'diagnostic only; not EVO-04 acceptance'}
    write_json(root / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--worktree', required=True, type=Path)
    parser.add_argument('--settings', type=Path)
    args = parser.parse_args()
    report = run(args.output_dir, args.config, args.worktree, settings_path=args.settings)
    return int(report['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())

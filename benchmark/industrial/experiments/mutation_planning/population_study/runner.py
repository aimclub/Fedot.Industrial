"""Freeze provenance before training; execute a deterministic resumable matrix."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import platform
from importlib.metadata import version
from time import monotonic, sleep

import psutil

from .data import file_hash, load_study_data
from .protocol import canonical, experiment_matrix, load_protocol
from .report import assess_study, render_report


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _source_snapshot():
    root = Path(__file__).resolve().parents[5]
    paths = [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('defaults.json'),
             Path(__file__).parent.parent / 'runner.py',
             root / 'benchmark/industrial/datasets/local_io.py',
             *sorted((root / 'fedot_ind/core/optimizer').rglob('*.py')),
             *sorted((root / 'fedot_ind/core/models').rglob('*.py')),
             *sorted((root / 'fedot_ind/core/operation').rglob('*.py')),
             root / 'fedot_ind/integration/fedot/extensions/catalog.json',
             root / 'fedot_ind/core/repository/data/default_operation_params.json']
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def freeze_run(protocol, root, worktree):
    from fedot_ind.integration.fedot.compat import audit_compatibility_worktree
    import fedot
    audit = audit_compatibility_worktree(worktree)
    if not Path(fedot.__file__).resolve().is_relative_to(Path(audit['worktree']) / 'fedot'):
        raise ValueError('Active FEDOT must belong to the explicitly audited copy')
    matrix = experiment_matrix(protocol)
    config = protocol.to_record()
    data_sources = {f"{s['name']}:{seed}": load_study_data(config, s, seed).source
                    for s in config['scenarios'] for seed in config['seeds']}
    manifest = {'protocol': config, 'protocol_sha256': protocol.digest, 'matrix': matrix,
                'data_sources': data_sources, 'implementation_sha256': _source_snapshot(),
                'fedot_compatibility': audit, 'python_executable': sys.executable,
                'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                'packages': {name: version(name) for name in
                                             ('numpy', 'torch', 'scikit-learn', 'dask', 'psutil', 'cloudpickle')}}}
    path = root / 'frozen_manifest.json'
    if path.exists():
        if canonical(json.loads(path.read_text(encoding='utf-8'))) != canonical(manifest):
            raise ValueError('Frozen protocol, code, data or environment changed; use a fresh output root')
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)
    return manifest


def execute_worker(job_file, timeout):
    environment = dict(os.environ)
    environment.update({key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    environment['CUDA_VISIBLE_DEVICES'] = '-1'
    environment['PYTHONHASHSEED'] = '0'
    command = [sys.executable, '-m',
               'benchmark.industrial.experiments.mutation_planning.population_study.worker', '--job', str(job_file)]
    deadline = monotonic() + timeout
    with job_file.with_name('run.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment) as process:
            try:
                while process.poll() is None:
                    if monotonic() >= deadline:
                        raise TimeoutError('Overall trial time limit exceeded')
                    try:
                        owner = psutil.Process(process.pid)
                        rss = owner.memory_info().rss
                        for child in owner.children(recursive=True):
                            try:
                                rss += child.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        break
                    if rss > 8 * 1024**3:
                        raise MemoryError('Trial exceeded the frozen 8 GiB process memory cap')
                    sleep(0.05)
            except BaseException:
                try:
                    owner = psutil.Process(process.pid)
                    for child in reversed(owner.children(recursive=True)):
                        child.kill()
                    owner.kill()
                except psutil.NoSuchProcess:
                    pass
                process.wait()
                raise
    return process.returncode


def run_study(output_dir, *, config_path=None, worktree=None, execute=False, limit=None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer or null')
    protocol = load_protocol(config_path)
    if not execute:
        return {'status': 'planned', 'protocol': protocol.to_record(), 'matrix': experiment_matrix(protocol)}
    if worktree is None:
        raise ValueError('Execution requires an audited compatibility worktree')
    root = Path(output_dir).resolve()
    manifest = freeze_run(protocol, root, worktree)
    records = []
    for job in manifest['matrix'][:limit]:
        if _source_snapshot() != manifest['implementation_sha256']:
            raise ValueError('Study implementation changed during execution')
        directory = root / f"run_{job['ordinal']:03}_{job['scenario']['name']}_{job['seed']}_{job['mode']}"
        directory.mkdir(exist_ok=True)
        result_path = directory / 'result.json'
        failure_path = directory / 'coordinator_failure.json'
        exit_path = directory / 'process.json'
        if not result_path.exists() and not failure_path.exists():
            write_json(directory / 'job.json', {'job': job, 'config': manifest['protocol']})
            print(f"START {job['ordinal'] + 1}/{len(manifest['matrix'])} {job['scenario']['name']} "
                  f"seed={job['seed']} {job['mode']}", flush=True)
            try:
                exit_code = execute_worker(directory / 'job.json', manifest['protocol']['run_seconds'])
                write_json(exit_path, {'exit_code': exit_code})
                if not result_path.exists():
                    raise RuntimeError(f'Worker exited {exit_code} without a result')
                if exit_code != 0:
                    raise RuntimeError(f'Worker exited {exit_code}; published result is not successful evidence')
            except (RuntimeError, TimeoutError, MemoryError) as error:
                write_json(failure_path, {**job, 'status': 'failed', 'error_type': type(error).__name__,
                                          'error_message': str(error)})
        record = json.loads((failure_path if failure_path.exists() else result_path).read_text(encoding='utf-8'))
        if record.get('status') == 'succeeded' and (
                not exit_path.exists() or json.loads(exit_path.read_text(encoding='utf-8'))['exit_code'] != 0):
            raise ValueError('Successful evidence requires a recorded zero process exit code')
        source_key = f"{job['scenario']['name']}:{job['seed']}"
        if record.get('status') == 'succeeded' and record.get('source') != manifest['data_sources'][source_key]:
            raise ValueError('Worker data provenance disagrees with the frozen manifest')
        records.append(record)
        print(f"DONE {job['ordinal']} {record['status']} loss={record.get('quality_loss')} "
              f"search={record.get('search_seconds')}", flush=True)
        write_json(root / 'records.json', records)
        report = assess_study(protocol, records)
        write_json(root / 'report.json', report)
        (root / 'report.md').write_text(render_report(report), encoding='utf-8')
    return assess_study(protocol, records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--compatibility-worktree', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--limit', type=int, help='Preflight only; cannot produce complete acceptance')
    args = parser.parse_args(argv)
    report = run_study(args.output_dir, config_path=args.config, worktree=args.compatibility_worktree,
                       execute=args.execute, limit=args.limit)
    print(json.dumps({key: report[key] for key in ('status', 'expected_runs', 'observed_runs') if key in report}))
    return int(report['status'] not in ('planned', 'accepted'))


if __name__ == '__main__':
    raise SystemExit(main())

"""Diagnostic CPU reruns of selected frozen trials, never full-study acceptance."""

import argparse
from importlib.metadata import distributions
import json
from pathlib import Path
import sys

from .data import file_hash, load_study_data
from .protocol import canonical, experiment_matrix, parse_protocol
from .runner import _source_snapshot, execute_worker, native_crash_message, write_json


def select_jobs(manifest, ordinals):
    """Select unchanged forecasting jobs from a validated original protocol."""
    if (not ordinals or any(type(value) is not int or value < 0 for value in ordinals)
            or len(set(ordinals)) != len(ordinals)):
        raise ValueError('Require unique nonnegative integer ordinals')
    protocol = parse_protocol(manifest['protocol'])
    if manifest['protocol_sha256'] != protocol.digest:
        raise ValueError('Original protocol digest does not match')
    matrix = experiment_matrix(protocol)
    if canonical(manifest['matrix']) != canonical(matrix):
        raise ValueError('Original frozen matrix differs from its protocol')
    selected = tuple(job for job in matrix if job['ordinal'] in ordinals)
    if len(selected) != len(ordinals) or any(job['scenario']['task'] != 'ts_forecasting' for job in selected):
        raise ValueError('Select only existing forecasting trials')
    return protocol, selected


def require_cpu_build(torch_module):
    if torch_module.version.cuda is not None or torch_module.cuda.is_available():
        raise RuntimeError('Diagnostic admission requires a CPU-only Torch build, not a masked CUDA build')


def _diagnostic_snapshot():
    root = Path(__file__).resolve().parents[5]
    integration = root / 'fedot_ind/integration/fedot'
    paths = sorted(path for path in integration.rglob('*')
                   if path.is_file() and path.suffix in ('.py', '.json', '.patch'))
    baseline = {Path(path).as_posix(): digest for path, digest in _source_snapshot().items()}
    return {**baseline, **{path.relative_to(root).as_posix(): file_hash(path) for path in paths}}


def recheck(source, output, ordinals, *, worktree=None, execute=False, repetitions=1):
    if type(repetitions) is not int or not 1 <= repetitions <= 3:
        raise ValueError('repetitions must be an integer from 1 to 3')
    source, output = Path(source).resolve(), Path(output).resolve()
    original_path = source / 'frozen_manifest.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    protocol, jobs = select_jobs(original, ordinals)
    plan = {'schema_version': 'industrial_forecast_recheck@1', 'status': 'planned',
            'purpose': 'diagnostic_only_not_full_acceptance', 'jobs': jobs, 'repetitions': repetitions,
            'original_manifest_sha256': file_hash(original_path), 'protocol_sha256': protocol.digest}
    if not execute:
        return plan
    import fedot
    import torch
    from fedot_ind.integration.fedot.compat import audit_compatibility_worktree

    require_cpu_build(torch)
    if worktree is None:
        raise ValueError('Execution requires an audited compatibility worktree')
    audit = audit_compatibility_worktree(worktree)
    if not Path(fedot.__file__).resolve().is_relative_to(Path(audit['worktree']) / 'fedot'):
        raise ValueError('Active FEDOT must belong to the audited worktree')
    if output == source or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Diagnostic output must be separate from original evidence')
    config = protocol.to_record()
    sources = {}
    for job in jobs:
        key = f"{job['scenario']['name']}:{job['seed']}"
        data_source = load_study_data(config, job['scenario'], job['seed']).source
        if data_source != original['data_sources'][key]:
            raise ValueError('Local data differs from the original trial')
        sources[key] = data_source
    implementation = _diagnostic_snapshot()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', {
        **plan, 'status': 'frozen', 'protocol': config, 'implementation_sha256': implementation,
        'python_executable': sys.executable, 'torch_version': torch.__version__,
        'torch_cuda_build': torch.version.cuda,
        'fedot_compatibility': audit, 'data_sources': sources,
        'packages': {dist.metadata['Name']: dist.version for dist in distributions()},
    })
    results = []
    for repeat in range(repetitions):
        for job in jobs:
            if _diagnostic_snapshot() != implementation:
                raise ValueError('Implementation changed during diagnostics')
            root = output / f"repeat_{repeat}_run_{job['ordinal']:03}"
            root.mkdir()
            write_json(root / 'job.json', {'job': job, 'config': config})
            print(f"START repeat={repeat} ordinal={job['ordinal']} {job['scenario']['name']} seed={job['seed']}",
                  flush=True)
            try:
                code = execute_worker(root / 'job.json', config['run_seconds'])
                write_json(root / 'process.json', {'exit_code': code})
                if _diagnostic_snapshot() != implementation or audit_compatibility_worktree(worktree) != audit:
                    raise ValueError('Implementation changed while the worker was running')
                result_file = root / 'result.json'
                result = json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else None
                fault = native_crash_message(root / 'run.log')
                if code != 0 or fault or result is None or result['status'] != 'succeeded':
                    detail = None if result is None else {
                        key: result.get(key) for key in ('status', 'error_type', 'error_message')}
                    raise RuntimeError(f'Unsuccessful worker: exit={code}, native_fault={fault}, result={detail}')
                if any(result.get(key) != value for key, value in job.items()):
                    raise ValueError('Worker trial identity differs from the frozen job')
                key = f"{job['scenario']['name']}:{job['seed']}"
                if result['source'] != sources[key]:
                    raise ValueError('Worker data provenance differs from the frozen diagnostic')
                record = {**job, 'repeat': repeat, 'status': 'succeeded',
                          'quality_loss': result['quality_loss'], 'search_seconds': result['search_seconds'],
                          'selected_graph': result['selected_graph'], 'result_sha256': file_hash(result_file)}
            except (RuntimeError, TimeoutError, MemoryError) as error:
                record = {**job, 'repeat': repeat, 'status': 'failed', 'error_type': type(error).__name__,
                          'error_message': str(error)}
            results.append(record)
            write_json(root / 'diagnostic.json', record)
            write_json(output / 'records.json', results)
            print(f"DONE ordinal={job['ordinal']} {record['status']}", flush=True)
    return {'status': 'diagnostics_complete', 'full_study_accepted': False, 'results': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ordinals', required=True, type=int, nargs='+')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--compatibility-worktree', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    result = recheck(args.source, args.output, args.ordinals, worktree=args.compatibility_worktree,
                     execute=args.execute, repetitions=args.repetitions)
    print(json.dumps(result))
    return int(any(record['status'] != 'succeeded' for record in result.get('results', ())))


if __name__ == '__main__':
    raise SystemExit(main())

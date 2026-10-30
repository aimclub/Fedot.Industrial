"""Resumable job-level supervisor with an explicit, evidence-backed smoke gate."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from ..preflight import file_digest
from .frozen_data import read_lock
from .plans import trial_plan


class ProtocolViolation(RuntimeError):
    pass


def checked_lock(bundle, identity):
    try:
        return read_lock(bundle, identity)
    except (ValueError, OSError) as error:
        raise ProtocolViolation(f'Frozen package or current source verification failed: {error}') from error


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def launcher_sources():
    paths = sorted([*Path(__file__).parent.glob('*.py'), *Path(__file__).parent.glob('*.json')])
    return {str(path.resolve()): file_digest(path) for path in paths}


def source_identity(sources):
    return sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def check_runtime(lock):
    import torch
    expected = lock['execution']['torch_flags']
    flags = dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                 deterministic=torch.are_deterministic_algorithms_enabled())
    if flags != expected:
        raise ProtocolViolation(f'Torch differs from frozen search runtime: {flags} != {expected}')
    return flags


def check_environment(expected):
    # Freeze captured distributions after FEDOT initialized pkg_resources' vendored paths.
    from importlib import import_module
    import_module('fedot')
    from importlib.metadata import distributions
    current = {dist.metadata['Name']: dist.version for dist in distributions()}
    if current != expected['packages'] or sys.version != expected['python']:
        raise ProtocolViolation('Dependency environment differs from frozen runtime')


def validate_result(directory, trial, lock):
    summary = json.loads((directory / 'summary.json').read_text('utf-8'))
    if (summary['source']['official_test_used'] or summary['test_loss'] is not None
            or summary['parameters'] != lock['training_reference']['operation_parameters']
            or not summary['owned_sessions_closed'] or summary['dask_sessions'] != 1
            or summary['torch_flags_before'] != lock['execution']['torch_flags']
            or summary['torch_flags_after'] != lock['execution']['torch_flags']):
        raise ProtocolViolation('Frozen result, resource or parameter contract violated')
    if trial.smoke and summary['outer_train_loss'] is not None:
        raise ProtocolViolation('Smoke evaluated the outer control')
    if not trial.smoke:
        import math
        if summary['outer_train_loss'] is None or not math.isfinite(summary['outer_train_loss']):
            raise ProtocolViolation('Full trial did not produce a finite TRAIN control evaluation')
    completed = json.loads((directory / 'completed_evaluations.json').read_text('utf-8'))
    selected = json.loads((directory / 'selected_graph.json').read_text('utf-8'))
    if not any(row['evaluation_context'].get('configuration') == selected for row in completed):
        raise ProtocolViolation('Selected configuration was not evaluated')
    with (directory / 'completed_evaluations.jsonl').open(encoding='utf-8') as stream:
        if [json.loads(line) for line in stream] != completed:
            raise ProtocolViolation('Incremental and final archives differ')
    records = [json.loads(line) for line in (directory / 'mutations.jsonl').read_text('utf-8').splitlines()]
    contract = json.loads((directory / 'arm_contract.json').read_text('utf-8'))
    if contract['arm'] != trial.arm or contract['policy'] != lock['policies'][trial.arm]:
        raise ProtocolViolation('Actual arm policy differs from frozen policy')
    if trial.arm == 'baseline':
        if 'rewrite_subgraph_mutation' in contract['actions']:
            raise ProtocolViolation('Baseline dispatcher contains rewrite')
        for row in records:
            planning = row.get('planning') or {}
            if (row['action'] == 'rewrite_subgraph_mutation' or 'rewrite' in planning.get('enabled_kinds', [])
                    or 'rewrite_subgraph_mutation' in row.get('agent_probabilities', {})):
                raise ProtocolViolation('Baseline dispatched or enumerated rewrite')
    failed = summary['diagnostics']['evaluations_failed']
    fraction = failed / max(1, failed + len(completed))
    smoke_passed = (summary['status'] == 'succeeded' and bool(records)
                    and fraction <= lock['acceptance_reference']['max_failed_evaluation_fraction'])
    return dict(status=summary['status'], evaluations=len(completed), failures=failed,
                generations=summary['completed_evolved_generations'], smoke_passed=smoke_passed,
                outer_train_loss=summary['outer_train_loss'])


def check_binding(root, bundle, seal, smoke):
    binding = json.loads((root / 'binding.json').read_text('utf-8'))
    if (binding['bundle'] != str(Path(bundle).resolve()) or binding['seal'] != seal or binding['smoke'] != smoke
            or binding['launcher_id'] != source_identity(launcher_sources())):
        raise ProtocolViolation('Run binding or launcher source changed')
    return binding


def worker(root, bundle, seal, index, *, smoke):
    from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
    from benchmark.industrial.experiments.evolution_parallelism.search import run_search
    from .study import FrozenStudy
    root, bundle = Path(root), Path(bundle)
    check_binding(root, bundle, seal, smoke)
    lock = checked_lock(bundle, seal)
    environment = json.loads((bundle / 'environment.json').read_text('utf-8'))
    check_environment(environment)
    check_runtime(lock)
    jobs = trial_plan(lock, smoke=smoke)
    if type(index) is not int or not 0 <= index < len(jobs):
        raise ProtocolViolation('Unknown job index')
    trial = jobs[index]
    directory = root / 'trials' / trial.key
    config = SearchConfig(trial.task, 'dask', generations=trial.generations,
                          workers=lock['protocol']['workers'], budget_seconds=trial.seconds, seed=trial.seed,
                          lifecycle='search', protocol='generations' if smoke else 'time', pilot=smoke,
                          native_threads=lock['protocol']['native_threads'])
    study = FrozenStudy(trial.arm, trial.task, pilot=smoke, bundle=str(bundle), lock=lock, trial=trial)
    run_search(config, directory, study=study)
    result = validate_result(directory, trial, lock)
    check_binding(root, bundle, seal, smoke)
    checked_lock(bundle, seal)
    atomic_json(directory / 'verified_result.json', result)
    return result


def receipt(directory, trial, exit_code, status, result=None):
    files = {path.relative_to(directory).as_posix(): file_digest(path)
             for path in sorted(directory.rglob('*')) if path.is_file()}
    return dict(trial=asdict(trial), exit_code=exit_code, status=status, result=result, files=files)


def verify_receipt(directory, value):
    observed = {path.relative_to(directory).as_posix() for path in directory.rglob('*') if path.is_file()}
    if observed != set(value['files']) or any(file_digest(directory / name) != expected
                                              for name, expected in value['files'].items()):
        raise ProtocolViolation('Completed trial artifacts changed; do not silently rerun')


def verify_gate(gate, bundle, seal, lock):
    gate = Path(gate)
    check_binding(gate, bundle, seal, True)
    rows = json.loads((gate / 'jobs.json').read_text('utf-8'))
    jobs = trial_plan(lock, smoke=True)
    if len(rows) != len(jobs):
        raise ProtocolViolation('Smoke matrix is incomplete')
    for row, trial in zip(rows, jobs):
        if (row['trial'] != asdict(trial) or row['exit_code'] != 0
                or not row['result'] or not row['result']['smoke_passed']):
            raise ProtocolViolation('Smoke did not pass; full series is blocked')
        verify_receipt(gate / 'trials' / trial.key, row)
    return file_digest(gate / 'jobs.json')


@contextmanager
def ownership(root):
    import psutil
    path = root / 'supervisor.lock'
    value = dict(pid=os.getpid(), create_time=psutil.Process().create_time())
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream)
    try:
        yield value
    finally:
        if json.loads(path.read_text('utf-8')) == value:
            path.unlink()


def supervise(root, bundle, seal, *, smoke, gate=None):
    from benchmark.industrial.experiments.evolution_parallelism.supervisor import run_guarded
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    if not smoke and gate is None:
        raise ProtocolViolation('Full series requires an explicit smoke gate')
    lock = checked_lock(bundle, seal)
    jobs = trial_plan(lock, smoke=smoke)
    gate_id = None if smoke else verify_gate(gate, bundle, seal, lock)
    root.mkdir(parents=True, exist_ok=True)
    with ownership(root) as owner:
        binding = dict(bundle=str(bundle), seal=seal, smoke=smoke, launcher_id=source_identity(launcher_sources()),
                       gate_id=gate_id, launch_permission='explicit user request and --execute',
                       created_at=datetime.now(timezone.utc).isoformat())
        if (root / 'binding.json').exists():
            previous = check_binding(root, bundle, seal, smoke)
            if previous['gate_id'] != gate_id:
                raise ProtocolViolation('Resume smoke gate differs from original')
        else:
            atomic_json(root / 'binding.json', binding)
            with ZipFile(root / 'launcher_sources.zip', 'x', compression=ZIP_DEFLATED) as archive:
                for name in launcher_sources():
                    archive.write(name, arcname=Path(name).name)
        (root / 'trials').mkdir(exist_ok=True)
        (root / 'receipts').mkdir(exist_ok=True)
        finished = []
        for trial in jobs:
            if (root / 'stop.request').exists():
                atomic_json(root / 'status.json', dict(status='stopped_between_jobs', completed=len(finished), **owner))
                return finished
            check_binding(root, bundle, seal, smoke)
            checked_lock(bundle, seal)
            directory = root / 'trials' / trial.key
            completed_path = root / 'receipts' / f'{trial.index:03d}.json'
            if completed_path.exists():
                row = json.loads(completed_path.read_text('utf-8'))
                if row['trial'] != asdict(trial):
                    raise ProtocolViolation('Saved trial differs from plan')
                verify_receipt(directory, row)
            else:
                if directory.exists():
                    raise ProtocolViolation(f'Interrupted trial requires inspection, not overwrite: {directory}')
                atomic_json(root / 'status.json', dict(status='running', current=asdict(trial), completed=len(finished),
                                                       total=len(jobs), updated_at=datetime.now(timezone.utc).isoformat(), **owner))
                command = [sys.executable, '-m', 'benchmark.industrial.experiments.mutation_quality.execution',
                           '--smoke' if smoke else '--run', '--execute', '--worker', str(trial.index),
                           '--bundle', str(bundle), '--seal', seal, '--output-dir', str(root)]
                print(json.dumps(dict(starting=asdict(trial))), flush=True)
                with (root / f'{trial.key}.log').open('x', encoding='utf-8') as stream:
                    try:
                        code = run_guarded(command, native_threads=lock['protocol']['native_threads'],
                                           search_seconds=trial.seconds, timeout_seconds=trial.process_seconds, output_stream=stream)
                        status = 'finished' if code == 0 else 'failed'
                    except subprocess.TimeoutExpired:
                        code, status = 3, 'timed_out'
                directory.mkdir(exist_ok=True)
                result = validate_result(directory, trial, lock) if code == 0 else None
                if result is not None:
                    status = result['status']
                row = receipt(directory, trial, code, status, result)
                atomic_json(completed_path, row)
            finished.append(row)
            atomic_json(root / 'jobs.json', finished)
            print(json.dumps(dict(completed=len(finished), total=len(jobs), trial=trial.key,
                                  exit_code=row['exit_code'], result=row['result'])), flush=True)
            if row['exit_code'] == 2 or (smoke and (row['exit_code'] != 0 or not row['result']['smoke_passed'])):
                atomic_json(root / 'status.json', dict(status='blocked',
                            completed=len(finished), current=asdict(trial), **owner))
                raise ProtocolViolation('Trial contract or smoke failed; stopping without parameter changes')
        atomic_json(root / 'status.json', dict(status='completed', completed=len(finished), total=len(jobs), **owner))
        return finished

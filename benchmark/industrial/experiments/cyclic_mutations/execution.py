"""Resumable B/D supervision; a new admission is mandatory before the 150 jobs."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from benchmark.industrial.experiments.mutation_quality.execution.plans import Trial
from benchmark.industrial.experiments.mutation_quality.execution.runner import (
    atomic_json, check_environment, check_runtime, ownership, terminal_status,
    receipt, verify_receipt, validate_result, ProtocolViolation,
)
from benchmark.industrial.experiments.mutation_quality.preflight import file_digest

from .freezing import base_directory, checked_lock
from .cleanup import confirmed_cleanup, failure_record
from .provenance import runtime_record


def must_stop(exit_code, admission, result):
    """Windows crash statuses can be unsigned values above the POSIX byte range."""
    return (exit_code < 0 or exit_code > 255 or exit_code == 2
            or (admission and (exit_code != 0 or not result or not result['smoke_passed'])))


def trials(lock, admission):
    rows = lock['jobs']
    if admission:
        rows = [r for name in lock['admission_scenarios'] for r in rows
                if r['scenario'] == name and r['seed'] == lock['protocol']['seeds'][0]]
    return tuple(Trial(i, r['scenario'], r['task'], r['arm'], r['seed'], r['search_seconds'],
                       r['minimum_generations'], r['full_process_seconds'], admission) for i, r in enumerate(rows))


def check_binding(root, bundle, identity, admission, base):
    value = json.loads((Path(root) / 'binding.json').read_text('utf-8'))
    if ((value['bundle'], value['seal'], value['admission'], value.get('base_bundle'))
            != (str(Path(bundle).resolve()), identity, admission, str(base))):
        raise ProtocolViolation('B/D execution binding differs')


def verify_runtime_record(directory, lock, identity):
    path = directory / 'runtime.json'
    if not path.is_file() or json.loads(path.read_text('utf-8')) != runtime_record(lock, identity):
        raise ProtocolViolation('Trial runtime evidence differs from the sealed procedure')


def validate(directory, trial, lock, identity):
    verify_runtime_record(directory, lock, identity)
    result = validate_result(directory, trial, lock)
    manifest = json.loads((directory / 'history/manifest.json').read_text('utf-8'))
    if not manifest['complete'] or manifest['raw_data_included']:
        raise ProtocolViolation('Native history was not fully checkpointed')
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    history = OptHistory.load(directory / 'history/opt_history.json')
    if history.generations_count != manifest['generations'] or not history.generations[-1][0].fitness.valid:
        raise ProtocolViolation('Native history round-trip failed')
    if not history.archive_history or history.archive_history[-1][0].uid != history.generations[-1][0].uid:
        raise ProtocolViolation('Native archive winner was lost during checkpointing')
    from golem.visualisation.opt_history.genealogical_path import trace_genealogical_path
    if not trace_genealogical_path(history.archive_history[-1][0], lambda left, right: 1):
        raise ProtocolViolation('Native lineage cannot be traced')
    if trial.arm == 'cyclic':
        phases = json.loads((directory / 'diagnostics/mutation_schedule.json').read_text('utf-8'))['events']
        if not any('_refine' in p['name'] for p in phases) or not any('_active' in p['name'] for p in phases):
            result['smoke_passed'] = False
            result['missing_cyclic_phase'] = True
    return result


def worker(root, bundle, identity, index, admission, *, base_bundle=None):
    from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
    from benchmark.industrial.experiments.evolution_parallelism.search import run_search
    from .study import GeneratorStudy
    lock = checked_lock(bundle, identity, base_bundle=base_bundle)
    base = base_directory(lock, base_bundle)
    check_binding(root, bundle, identity, admission, base)
    check_environment(json.loads((Path(bundle) / 'environment.json').read_text('utf-8')))
    check_runtime(lock)
    plan = trials(lock, admission)
    if type(index) is not int or not 0 <= index < len(plan):
        raise ProtocolViolation('Unknown B/D trial index')
    trial = plan[index]
    directory = Path(root) / 'trials' / trial.key
    config = SearchConfig(trial.task, 'dask', generations=trial.generations, workers=4, budget_seconds=trial.seconds,
                          seed=trial.seed, lifecycle='search', protocol='time', pilot=admission, native_threads=1)
    study = GeneratorStudy(trial.arm, trial.task, pilot=admission,
                           bundle=str(base), lock=lock, trial=trial)
    from fedot_ind.core.optimizer.evaluation.dask_executor import DaskCleanupError
    try:
        run_search(config, directory, study=study)
    except DaskCleanupError as error:
        atomic_json(directory / 'worker_failure.json', failure_record(error))
        raise ProtocolViolation('Dask cleanup was not confirmed; stop the series') from error
    from .history import render_native_history
    renders = render_native_history(directory / 'history/opt_history.json',
                                    directory / 'native_visuals', animations=False)
    if any(row['status'] != 'succeeded' for row in renders if row['plot'] in ('fitness_line', 'best_graph')):
        raise ProtocolViolation('Required native result plots failed')
    checked_lock(bundle, identity, base_bundle=base)
    atomic_json(directory / 'runtime.json', runtime_record(lock, identity))
    result = validate(directory, trial, lock, identity)
    atomic_json(directory / 'verified_result.json', result)


def verify_admission(root, bundle, identity, lock, base):
    root = Path(root)
    check_binding(root, bundle, identity, True, base)
    jobs = json.loads((root / 'jobs.json').read_text('utf-8'))
    expected = trials(lock, True)
    if len(jobs) != len(expected):
        raise ProtocolViolation('New B/D admission is incomplete')
    for row, trial in zip(jobs, expected):
        if row['trial'] != asdict(trial) or row['exit_code'] != 0 or not row['result']['smoke_passed']:
            raise ProtocolViolation('New B/D admission failed')
        directory = root / 'trials' / trial.key
        verify_receipt(directory, row)
        verify_runtime_record(directory, lock, identity)
    return file_digest(root / 'jobs.json')


def supervise(root, bundle, identity, admission, gate=None, *, resume_after_cleanup=None, base_bundle=None):
    from benchmark.industrial.experiments.evolution_parallelism.supervisor import run_guarded
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    lock = checked_lock(bundle, identity, base_bundle=base_bundle)
    if not admission and gate is None:
        raise ProtocolViolation('Full B/D series requires its own successful admission')
    plan = trials(lock, admission)
    if resume_after_cleanup is not None:
        if admission or type(resume_after_cleanup) is not int or not 0 <= resume_after_cleanup < len(plan):
            raise ProtocolViolation('Cleanup confirmation must identify a blocked full-series trial')
        blocked = root / 'receipts' / f'{resume_after_cleanup:03d}.json'
        if not blocked.is_file() or json.loads(blocked.read_text('utf-8'))['exit_code'] != 2:
            raise ProtocolViolation('No blocked receipt at the requested trial index')
    base = base_directory(lock, base_bundle)
    gate_id = None if admission else verify_admission(gate, bundle, identity, lock, base)
    root.mkdir(parents=True, exist_ok=True)
    with ownership(root) as owner, terminal_status(root):
        binding = dict(bundle=str(bundle), seal=identity, admission=admission, gate_id=gate_id, base_bundle=str(base))
        if (root / 'binding.json').exists():
            if json.loads((root / 'binding.json').read_text('utf-8')) != binding:
                raise ProtocolViolation('Changed admission or run binding')
        else:
            atomic_json(root / 'binding.json', binding)
        (root / 'trials').mkdir(exist_ok=True)
        (root / 'receipts').mkdir(exist_ok=True)
        finished = []
        for trial in plan:
            if (root / 'stop.request').exists():
                atomic_json(root / 'status.json', dict(status='stopped_between_jobs', completed=len(finished), **owner))
                return
            checked_lock(bundle, identity, base_bundle=base)
            path, directory = root / 'receipts' / f'{trial.index:03d}.json', root / 'trials' / trial.key
            replaying = path.exists()
            if replaying:
                row = json.loads(path.read_text('utf-8'))
                if row['trial'] != asdict(trial):
                    raise ProtocolViolation('Saved B/D trial differs')
                verify_receipt(directory, row)
                if row['exit_code'] == 0:
                    verify_runtime_record(directory, lock, identity)
            else:
                if resume_after_cleanup is not None and trial.index <= resume_after_cleanup:
                    raise ProtocolViolation('Cleanup confirmation requires an intact receipt prefix')
                if directory.exists():
                    raise ProtocolViolation('Unfinished trial requires inspection, never overwrite')
                atomic_json(root / 'status.json', dict(status='running', current=asdict(trial),
                                                       completed=len(finished), total=len(plan), **owner))
                command = [sys.executable, '-m', 'benchmark.industrial.experiments.cyclic_mutations',
                           '--admission' if admission else '--run', '--bundle', str(bundle), '--seal', identity,
                           '--output-dir', str(root), '--execute', '--worker', str(trial.index), '--base-bundle', str(base)]
                print(json.dumps(dict(starting=asdict(trial))), flush=True)
                with (root / f'{trial.key}.log').open('x', encoding='utf-8') as stream:
                    try:
                        code = run_guarded(command, native_threads=1, timeout_seconds=1200,
                                           search_seconds=trial.seconds, output_stream=stream)
                    except subprocess.TimeoutExpired:
                        code = 3
                directory.mkdir(exist_ok=True)
                result = validate(directory, trial, lock, identity) if code == 0 else None
                row = receipt(directory, trial, code, 'failed' if result is None else result['status'], result)
                atomic_json(path, row)
            finished.append(row)
            atomic_json(root / 'jobs.json', finished)
            print(json.dumps(dict(completed=len(finished), total=len(plan), trial=trial.key,
                                  exit_code=row['exit_code'], result=row['result'])), flush=True)
            if must_stop(row['exit_code'], admission, row['result']):
                atomic_json(root / 'status.json', dict(status='blocked', completed=len(finished), **owner))
                confirmation = root / 'cleanup_confirmations' / path.name
                if not admission and replaying and (resume_after_cleanup == trial.index or confirmation.is_file()):
                    proof = confirmed_cleanup(directory, path, row)
                    if confirmation.is_file():
                        if json.loads(confirmation.read_text('utf-8')) != proof:
                            raise ProtocolViolation('Stored cleanup confirmation differs from verified evidence')
                    else:
                        confirmation.parent.mkdir(exist_ok=True)
                        atomic_json(confirmation, proof)
                    continue
                raise ProtocolViolation('B/D trial failed; full series remains blocked')
        atomic_json(root / 'status.json', dict(status='completed', completed=len(finished), total=len(plan), **owner))

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

from .freezing import checked_lock


def trials(lock, admission):
    rows = lock['jobs']
    if admission:
        rows = [r for r in rows if r['scenario'] in lock['admission_scenarios']
                and r['seed'] == lock['protocol']['seeds'][0]]
    return tuple(Trial(i, r['scenario'], r['task'], r['arm'], r['seed'], r['search_seconds'],
                       r['minimum_generations'], r['full_process_seconds'], admission) for i, r in enumerate(rows))


def check_binding(root, bundle, identity, admission):
    value = json.loads((Path(root) / 'binding.json').read_text('utf-8'))
    if (value['bundle'], value['seal'], value['admission']) != (str(Path(bundle).resolve()), identity, admission):
        raise ProtocolViolation('B/D execution binding differs')


def validate(directory, trial, lock):
    result = validate_result(directory, trial, lock)
    manifest = json.loads((directory / 'history/manifest.json').read_text('utf-8'))
    if not manifest['complete'] or manifest['raw_data_included']:
        raise ProtocolViolation('Native history was not fully checkpointed')
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    history = OptHistory.load(directory / 'history/opt_history.json')
    if history.generations_count != manifest['generations'] or not history.generations[-1][0].fitness.valid:
        raise ProtocolViolation('Native history round-trip failed')
    if trial.arm == 'cyclic':
        phases = json.loads((directory / 'diagnostics/mutation_schedule.json').read_text('utf-8'))['events']
        if not any('_refine' in p['name'] for p in phases) or not any('_active' in p['name'] for p in phases):
            result['smoke_passed'] = False
            result['missing_cyclic_phase'] = True
    return result


def worker(root, bundle, identity, index, admission):
    from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
    from benchmark.industrial.experiments.evolution_parallelism.search import run_search
    from .study import GeneratorStudy
    lock = checked_lock(bundle, identity)
    check_binding(root, bundle, identity, admission)
    check_environment(json.loads((Path(bundle) / 'environment.json').read_text('utf-8')))
    check_runtime(lock)
    trial = trials(lock, admission)[index]
    directory = Path(root) / 'trials' / trial.key
    config = SearchConfig(trial.task, 'dask', generations=trial.generations, workers=4, budget_seconds=trial.seconds,
                          seed=trial.seed, lifecycle='search', protocol='time', pilot=admission, native_threads=1)
    study = GeneratorStudy(trial.arm, trial.task, pilot=admission,
                           bundle=lock['base']['directory'], lock=lock, trial=trial)
    from fedot_ind.core.optimizer.evaluation.dask_executor import DaskCleanupError
    try:
        run_search(config, directory, study=study)
    except DaskCleanupError as error:
        raise ProtocolViolation('Dask cleanup was not confirmed; stop the series') from error
    from .history import render_native_history
    renders = render_native_history(directory / 'history/opt_history.json', directory / 'native_visuals', animations=False)
    if any(row['status'] != 'succeeded' for row in renders if row['plot'] in ('fitness_line', 'best_graph')):
        raise ProtocolViolation('Required native result plots failed')
    result = validate(directory, trial, lock)
    checked_lock(bundle, identity)
    atomic_json(directory / 'verified_result.json', result)


def verify_admission(root, bundle, identity, lock):
    root = Path(root)
    check_binding(root, bundle, identity, True)
    jobs = json.loads((root / 'jobs.json').read_text('utf-8'))
    expected = trials(lock, True)
    if len(jobs) != len(expected):
        raise ProtocolViolation('New B/D admission is incomplete')
    for row, trial in zip(jobs, expected):
        if row['trial'] != asdict(trial) or row['exit_code'] != 0 or not row['result']['smoke_passed']:
            raise ProtocolViolation('New B/D admission failed')
        verify_receipt(root / 'trials' / trial.key, row)
    return file_digest(root / 'jobs.json')


def supervise(root, bundle, identity, admission, gate=None):
    from benchmark.industrial.experiments.evolution_parallelism.supervisor import run_guarded
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    lock = checked_lock(bundle, identity)
    if not admission and gate is None:
        raise ProtocolViolation('Full B/D series requires its own successful admission')
    gate_id = None if admission else verify_admission(gate, bundle, identity, lock)
    root.mkdir(parents=True, exist_ok=True)
    with ownership(root) as owner, terminal_status(root):
        binding = dict(bundle=str(bundle), seal=identity, admission=admission, gate_id=gate_id)
        if (root / 'binding.json').exists():
            if json.loads((root / 'binding.json').read_text('utf-8')) != binding:
                raise ProtocolViolation('Changed admission or run binding')
        else:
            atomic_json(root / 'binding.json', binding)
        (root / 'trials').mkdir(exist_ok=True)
        (root / 'receipts').mkdir(exist_ok=True)
        finished, plan = [], trials(lock, admission)
        for trial in plan:
            if (root / 'stop.request').exists():
                atomic_json(root / 'status.json', dict(status='stopped_between_jobs', completed=len(finished), **owner))
                return
            checked_lock(bundle, identity)
            path, directory = root / 'receipts' / f'{trial.index:03d}.json', root / 'trials' / trial.key
            if path.exists():
                row = json.loads(path.read_text('utf-8'))
                if row['trial'] != asdict(trial):
                    raise ProtocolViolation('Saved B/D trial differs')
                verify_receipt(directory, row)
            else:
                if directory.exists():
                    raise ProtocolViolation('Unfinished trial requires inspection, never overwrite')
                atomic_json(root / 'status.json', dict(status='running', current=asdict(trial),
                                                       completed=len(finished), total=len(plan), **owner))
                command = [sys.executable, '-m', 'benchmark.industrial.experiments.cyclic_mutations',
                           '--admission' if admission else '--run', '--bundle', str(bundle), '--seal', identity,
                           '--output-dir', str(root), '--execute', '--worker', str(trial.index)]
                print(json.dumps(dict(starting=asdict(trial))), flush=True)
                with (root / f'{trial.key}.log').open('x', encoding='utf-8') as stream:
                    try:
                        code = run_guarded(command, native_threads=1, timeout_seconds=1200,
                                           search_seconds=trial.seconds, output_stream=stream)
                    except subprocess.TimeoutExpired:
                        code = 3
                directory.mkdir(exist_ok=True)
                result = validate(directory, trial, lock) if code == 0 else None
                row = receipt(directory, trial, code, 'failed' if result is None else result['status'], result)
                atomic_json(path, row)
            finished.append(row)
            atomic_json(root / 'jobs.json', finished)
            print(json.dumps(dict(completed=len(finished), total=len(plan), trial=trial.key,
                                  exit_code=row['exit_code'], result=row['result'])), flush=True)
            if row['exit_code'] == 2 or row['exit_code'] < 0 or (admission and (
                    row['exit_code'] != 0 or not row['result']['smoke_passed'])):
                atomic_json(root / 'status.json', dict(status='blocked', completed=len(finished), **owner))
                raise ProtocolViolation('B/D trial failed; full series remains blocked')
        atomic_json(root / 'status.json', dict(status='completed', completed=len(finished), total=len(plan), **owner))

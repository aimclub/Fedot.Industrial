"""Owned, sequential Dask searches with immutable source snapshots and resumable jobs."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .training import StagedTrainingStudy, load_training_protocol


def implementation_paths():
    import fedot
    study = StagedTrainingStudy('fixed', 'classification')
    return sorted(set([*study.implementation_paths(), *Path('fedot_ind/core/optimizer').rglob('*.py'),
                       *Path('fedot_ind').rglob('*.py'), *Path('fedot_ind').rglob('*.json'),
                       *Path(fedot.__file__).parent.rglob('*.py'), *Path(fedot.__file__).parent.rglob('*.json'),
                       *Path('benchmark/industrial/experiments/evolution_parallelism').glob('*.py'),
                       Path('benchmark/industrial/experiments/evolution_parallelism/defaults.json'),
                       Path('fedot_ind/core/repository/data/default_operation_params.json'),
                       Path('benchmark/industrial/experiments/mutation_planning/population_study/preparation.py')]))


def run_trial(root, task, mode, seed, *, pilot=False):
    from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
    from benchmark.industrial.experiments.evolution_parallelism.search import run_search, write_json
    protocol = load_training_protocol()
    if seed not in protocol['seeds'] or task not in protocol['tasks']:
        raise ValueError('Trial is outside the approved task/seed matrix')
    seconds = protocol['pilot_seconds' if pilot else 'budget_seconds']
    config = SearchConfig(task, 'dask', workers=protocol['workers'], budget_seconds=seconds, seed=seed,
                          generations=1 if pilot else protocol['minimum_generations'], lifecycle='search',
                          protocol='time', pilot=pilot, native_threads=protocol['native_threads'])
    result = run_search(config, root, study=StagedTrainingStudy(mode, task, pilot))
    write_json(Path(root) / 'training_request.json', {'protocol': protocol, 'mode': mode, 'request': asdict(config)})
    if result['test_loss'] is not None or result['source']['official_test_used'] or result['dask_sessions'] != 1:
        raise AssertionError('TRAIN-only single-session protocol violated')
    if not result['owned_sessions_closed']:
        raise AssertionError('An owned Dask client was not closed')
    completed = json.loads((Path(root) / 'completed_evaluations.json').read_text('utf-8'))
    selected = json.loads((Path(root) / 'selected_graph.json').read_text('utf-8'))
    matches = [row for row in completed if row['evaluation_context'].get('configuration') == selected]
    if not matches:
        raise AssertionError('Selected model configuration differs from its evaluated configuration')
    return result


def job_matrix(protocol, pilot=False):
    jobs = []
    for task_index, task in enumerate(protocol['tasks']):
        seeds = protocol['seeds'][:1] if pilot else protocol['seeds']
        for seed_index, seed in enumerate(seeds):
            modes = ['staged'] if pilot else ['fixed', 'staged']
            if (task_index + seed_index) % 2:
                modes.reverse()
            jobs.extend((task, mode, seed) for mode in modes)
    return tuple(jobs)


def freeze(root):
    root = Path(root)
    path = root / 'frozen_sources.json'
    if path.exists():
        frozen = json.loads(path.read_text('utf-8'))
    else:
        frozen = {str(p.resolve()): {'sha256': sha256(p.read_bytes()).hexdigest(), 'source': p.read_text('utf-8')}
                  for p in implementation_paths()}
        with path.open('x', encoding='utf-8') as stream:
            json.dump(frozen, stream, indent=2)
            stream.write('\n')
    if any(sha256(Path(name).read_bytes()).hexdigest() != item['sha256'] for name, item in frozen.items()):
        raise RuntimeError('Frozen source changed; do not mix procedures in one comparison')
    return frozen


def run_matrix(root, *, pilot=False):
    from benchmark.industrial.experiments.evolution_parallelism.search import write_json
    from benchmark.industrial.experiments.evolution_parallelism.supervisor import run_guarded
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_training_protocol()
    frozen = freeze(root)
    write_json(root / 'protocol.json', {'protocol': protocol, 'pilot': pilot, 'order': job_matrix(protocol, pilot)})
    jobs = []
    for task, mode, seed in job_matrix(protocol, pilot):
        if {str(p.resolve()) for p in implementation_paths()} != set(frozen):
            raise RuntimeError('Source membership changed between trials')
        if any(sha256(Path(name).read_bytes()).hexdigest() != item['sha256'] for name, item in frozen.items()):
            raise RuntimeError('Frozen source changed between trials')
        directory = root / f'{task}_{seed}_{mode}'
        previous = directory / 'supervisor.json'
        if previous.exists():
            row = json.loads(previous.read_text('utf-8'))
            jobs.append(row)
            continue
        if directory.exists():
            raise FileExistsError(f'Unfinished trial must be inspected, not overwritten: {directory}')
        launch = root / f'{task}_{seed}_{mode}.log'
        command = [sys.executable, '-m', 'benchmark.industrial.experiments.staged_mutations', '--train', '--worker',
                   '--execute', '--mode', mode, '--task', task, '--seed', str(seed), '--output-dir', str(directory)]
        if pilot:
            command.append('--pilot')
        seconds = protocol['pilot_seconds' if pilot else 'budget_seconds']
        start = perf_counter()
        print(json.dumps({'starting': [task, mode, seed], 'pilot': pilot}), flush=True)
        with launch.open('x', encoding='utf-8') as log:
            try:
                code = run_guarded(command, native_threads=protocol['native_threads'], search_seconds=seconds,
                                   timeout_seconds=protocol['full_process_seconds'], output_stream=log)
                status = 'finished' if code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                code, status = 2, 'timed_out'
        directory.mkdir(exist_ok=True)
        row = {'task': task, 'mode': mode, 'seed': seed, 'exit_code': code, 'status': status,
               'end_to_end_seconds': perf_counter() - start}
        write_json(directory / 'supervisor.json', row)
        jobs.append(row)
        write_json(root / 'jobs.json', jobs)
        print(json.dumps(row), flush=True)
        if pilot and code:
            raise RuntimeError(f'Pilot failed; inspect {launch}')
    return jobs

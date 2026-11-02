"""Frozen TRAIN-only A/B jobs with explicit, bounded subprocess ownership."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from .study import MutationStudy, load_protocol


def run_trial(root, task, arm, seed):
    from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
    from benchmark.industrial.experiments.evolution_parallelism.search import run_search
    protocol = load_protocol()
    config = SearchConfig(task, 'dask', budget_seconds=protocol['budget_seconds'], seed=seed,
                          lifecycle='search', protocol='time')
    result = run_search(config, root, study=MutationStudy(arm, task))
    (Path(root) / 'study.json').write_text(json.dumps(dict(protocol=protocol, arm=arm, request=asdict(config)),
                                                      indent=2) + '\n', encoding='utf-8')
    if result['test_loss'] is not None or result['source']['official_test_used']:
        raise AssertionError('Official TEST must not be used by this study')
    return result


def run_matrix(root, generation_root):
    from benchmark.industrial.experiments.evolution_parallelism.supervisor import run_guarded
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol()
    generation = json.loads((Path(generation_root) / 'generation.json').read_text('utf-8'))
    if generation['training_calls'] != 0:
        raise ValueError('Generation audit unexpectedly used training')
    gates = {}
    for task in protocol['training_tasks']:
        initial = set(generation['initial_configuration_ids'][task])
        observed = {arm: set().union(*(set(row['configuration_ids']) for row in generation['rows']
                    if row['task'] == task and row['arm'] == arm)) for arm in protocol['arms']}
        gates[task] = len(observed['structural'] - (observed['baseline'] | initial)) > 0
    (root / 'protocol.json').write_text(json.dumps(dict(protocol=protocol, generation_gates=gates), indent=2) + '\n', encoding='utf-8')
    if not all(gates.values()):
        raise ValueError('Generation did not open new admissible configurations; training is not justified')
    from hashlib import sha256
    files = MutationStudy('baseline', 'classification').implementation_paths()
    files.extend(Path('fedot_ind/core/optimizer').rglob('*.py'))
    files.extend(Path('benchmark/industrial/experiments/evolution_parallelism').glob('*.py'))
    files.extend([Path('benchmark/industrial/experiments/evolution_parallelism/defaults.json'),
                  Path('benchmark/industrial/experiments/mutation_planning/population_study/preparation.py'),
                  Path('fedot_ind/core/repository/data/default_operation_params.json')])
    frozen = {str(p.resolve()): dict(sha256=sha256(p.read_bytes()).hexdigest(), source=p.read_text('utf-8'))
              for p in files}
    (root / 'frozen_sources.json').write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
    from time import perf_counter
    jobs = []
    # Interleave A/B order by seed and task, fixed before observing any result.
    for task_index, task in enumerate(protocol['training_tasks']):
        for seed_index, seed in enumerate(protocol['seeds']):
            arms = list(protocol['arms'])
            if (task_index + seed_index) % 2:
                arms.reverse()
            for arm in arms:
                if any(sha256(Path(p).read_bytes()).hexdigest() != item['sha256'] for p, item in frozen.items()):
                    raise RuntimeError('Frozen implementation changed; do not continue with mixed procedures')
                directory = root / f'{task}_{seed}_{arm}'
                command = [sys.executable, '-m', 'benchmark.industrial.experiments.mutation_diversity',
                           '--worker', '--execute', '--task', task, '--arm', arm, '--seed', str(seed), '--output-dir', str(directory)]
                start = perf_counter()
                try:
                    code = run_guarded(command, native_threads=1, search_seconds=protocol['budget_seconds'])
                    status = 'finished' if code == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    code, status = 2, 'timed_out'
                row = dict(task=task, arm=arm, seed=seed, exit_code=code, status=status,
                           end_to_end_seconds=perf_counter() - start)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'supervisor.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
                jobs.append(row)
                (root / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n', encoding='utf-8')
                print(json.dumps(row), flush=True)
    return jobs

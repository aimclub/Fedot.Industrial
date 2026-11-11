"""Bounded sequential technical matrix; no full-study launch capability."""

from dataclasses import asdict
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from .config import Arm, load_protocol


def technical_settings():
    value = json.loads(Path(__file__).with_name('technical.json').read_text('utf-8'))
    if (any(type(value.get(key)) is not int for key in (
            'seed', 'generations', 'search_seconds', 'full_process_seconds', 'maximum_trials'))
            or value.get('schema_version') != 'industrial_mutation_quality_technical@1'
            or value.get('scenarios') != ['ECG200', 'NaturalGasPricesSentiment', 'Airline']
            or value.get('seed') != 42 or value.get('generations') != 1
            or value.get('search_seconds') != 150 or value.get('full_process_seconds') != 420
            or value.get('maximum_trials') != 9):
        raise ValueError('Technical matrix exceeds the approved pilot configuration')
    return value


def technical_jobs():
    settings = technical_settings()
    return tuple((name, list(Arm)[(index + offset) % len(Arm)].value, settings['seed'])
                 for index, name in enumerate(settings['scenarios']) for offset in range(len(Arm)))


def verify_trial(root, arm):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_text('utf-8'))
    contract = json.loads((root / 'arm_contract.json').read_text('utf-8'))
    if contract['arm'] != arm or summary['status'] != 'succeeded':
        raise AssertionError('Technical trial did not complete its requested arm and generation')
    if (summary['test_loss'] is not None or summary['outer_train_loss'] is not None
            or summary['source']['official_test_used'] or summary['dask_sessions'] != 1
            or not summary['owned_sessions_closed']):
        raise AssertionError('TRAIN-only single owned-session contract violated')
    if summary['torch_flags_before'] != summary['torch_flags_after']:
        raise AssertionError('Torch settings changed')
    mutations = [json.loads(line) for line in (root / 'mutations.jsonl').read_text('utf-8').splitlines()]
    if not mutations:
        raise AssertionError('No evolutionary mutations were observed')
    if Arm(arm) is Arm.BASELINE:
        for row in mutations:
            planning = row.get('planning') or {}
            if (row['action'] == 'rewrite_subgraph_mutation'
                    or 'rewrite_subgraph_mutation' in row.get('agent_probabilities', {})
                    or 'rewrite' in planning.get('enabled_kinds', [])
                    or set(planning.get('family_counts', {})) & {
                        'rewrite_chain', 'fork_feature_branch', 'fork_forecast_branch', 'prune_feature_branch'}):
                raise AssertionError('Baseline enumerated or dispatched a forbidden structural rewrite')
    completed = json.loads((root / 'completed_evaluations.json').read_text('utf-8'))
    selected = json.loads((root / 'selected_graph.json').read_text('utf-8'))
    if not any(item['evaluation_context'].get('configuration') == selected for item in completed):
        raise AssertionError('Selected configuration was not evaluated')
    return dict(arm=arm, scenario=summary['source']['dataset'], mutations=len(mutations),
                action_counts=dict(Counter(row['action'] for row in mutations)),
                completed_evaluations=summary['completed_evaluations'],
                generations=summary['completed_evolved_generations'],
                actions=contract['actions'], torch_flags=summary['torch_flags_after'],
                source_id=sha256(json.dumps(summary['source'], sort_keys=True, allow_nan=False).encode()).hexdigest(),
                parameters_id=sha256(json.dumps(summary['parameters'],
                                     sort_keys=True, allow_nan=False).encode()).hexdigest(),
                owned_sessions_closed=summary['owned_sessions_closed'], official_test_used=False)


def run_trial(root, scenario_name, arm, seed):
    if (scenario_name, arm, seed) not in technical_jobs():
        raise ValueError('Trial is outside the bounded technical matrix')
    from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
    from benchmark.industrial.experiments.evolution_parallelism.search import run_search, write_json
    from .study import QualityStudy
    settings, protocol = technical_settings(), load_protocol()
    scenario = next(row for row in protocol.scenarios if row.name == scenario_name)
    config = SearchConfig(scenario.task, 'dask', generations=settings['generations'], workers=protocol.workers,
                          budget_seconds=settings['search_seconds'], seed=seed, lifecycle='search',
                          protocol='generations', pilot=True, native_threads=protocol.native_threads)
    study = QualityStudy(arm, scenario.task, pilot=True, scenario_name=scenario_name)
    run_search(config, root, study=study)
    result = verify_trial(root, arm)
    write_json(Path(root) / 'technical_verification.json', result)
    write_json(Path(root) / 'quality_request.json', dict(request=asdict(config), protocol=asdict(protocol)))
    return result


def freeze_sources():
    from benchmark.industrial.experiments.staged_mutations.training_runner import implementation_paths
    paths = set(implementation_paths()) | set(Path(__file__).parent.glob('*.py')) | set(
        Path(__file__).parent.glob('*.json'))
    return {str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def run_technical(root):
    from benchmark.industrial.experiments.evolution_parallelism.search import write_json
    from benchmark.industrial.experiments.evolution_parallelism.supervisor import run_guarded
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    settings = technical_settings()
    frozen = freeze_sources()
    write_json(root / 'frozen_sources.json', frozen)
    write_json(root / 'technical_plan.json', dict(settings=settings, jobs=technical_jobs(), large_series=False))
    results = []
    shared_inputs = {}
    for name, arm, seed in technical_jobs():
        if frozen != freeze_sources():
            raise RuntimeError('Source snapshot changed between technical trials')
        directory = root / f'{name}_{arm}_{seed}'
        command = [sys.executable, '-m', 'benchmark.industrial.experiments.mutation_quality',
                   '--technical', '--worker', '--execute', '--scenario', name, '--arm', arm,
                   '--output-dir', str(directory)]
        print(json.dumps(dict(starting=[name, arm, seed])), flush=True)
        with (root / f'{name}_{arm}.log').open('x', encoding='utf-8') as log:
            try:
                code = run_guarded(command, native_threads=load_protocol().native_threads,
                                   search_seconds=settings['search_seconds'],
                                   timeout_seconds=settings['full_process_seconds'], output_stream=log)
                status = 'finished' if code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                code, status = 2, 'timed_out'
        row = dict(scenario=name, arm=arm, seed=seed, exit_code=code, status=status)
        if not code:
            row['verification'] = verify_trial(directory, arm)
            identity = (row['verification']['source_id'], row['verification']['parameters_id'],
                        row['verification']['torch_flags'])
            if name in shared_inputs and shared_inputs[name] != identity:
                raise AssertionError('Data, initial parameters or Torch settings differ between arms')
            shared_inputs[name] = identity
        results.append(row)
        write_json(root / 'jobs.json', results)
        print(json.dumps(row), flush=True)
        if code:
            raise RuntimeError(f'Technical trial failed, stopping the matrix: {directory}')
    if frozen != freeze_sources():
        raise RuntimeError('Source snapshot changed during technical trials')
    return results

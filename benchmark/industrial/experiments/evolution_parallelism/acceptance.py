"""Bounded paired acceptance; never tunes models or uses TEST to select them."""
import argparse
from collections import Counter
import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys


def validate_pair(root):
    def read(run, name):
        return json.loads((root / run / name).read_text(encoding='utf-8'))

    seq, dask = (read(run, 'summary.json') for run in ('sequential', 'dask'))

    def pool(run):
        return Counter((row['evaluation_context']['configuration_id'], tuple(row['fitness']))
                       for row in read(run, 'completed_evaluations.json'))

    def attempts(run):
        rows = [json.loads(line) for path in (root / run / 'traces').glob('evaluations_*.jsonl')
                for line in path.read_text('utf-8').splitlines() if line.strip()]
        return Counter((row['configuration_id'], row['valid'], row['fitness']) for row in rows)

    checks = {
        'both_complete': all(row['status'] == 'succeeded' and row['minimum_generations_met'] for row in (seq, dask)),
        'same_training_data': seq['source']['train_sha256'] == dask['source']['train_sha256'],
        'same_folds': seq['source']['folds'] == dask['source']['folds'],
        'test_not_used_for_search': all(not row['source']['test_used_for_search'] for row in (seq, dask)),
        'same_evaluated_pool_and_fitness': pool('sequential') == pool('dask'),
        'same_selected_graph': read('sequential', 'selected_graph.json') == read('dask', 'selected_graph.json'),
        'same_final_metric': seq['test_loss'] == dask['test_loss'],
        'torch_unchanged': all(row['torch_flags_before'] == row['torch_flags_after'] for row in (seq, dask)),
        'one_owned_session_closed': dask['dask_sessions'] == 1 and dask['owned_sessions_closed'],
        'same_actual_attempts': bool(attempts('sequential')) and attempts('sequential') == attempts('dask'),
        'same_existing_fitness_reuse': seq['diagnostics'].get('evaluations_reused', 0)
        == dask['diagnostics'].get('evaluations_reused', 0),
    }
    result = {'checks': checks, 'passed': all(checks.values()), 'runs': {
        'sequential': {key: seq[key] for key in ('search_wall_seconds', 'wall_seconds', 'test_loss', 'phases')},
        'dask': {key: dask[key] for key in ('search_wall_seconds', 'wall_seconds', 'test_loss', 'phases')}},
        'scope': 'Airline, one seed, 15 evolved generations; not a statistical speed claim'}
    result['supervisor_seconds'] = {run: read(run, 'supervisor.json')['end_to_end_seconds']
                                    for run in ('sequential', 'dask')}
    result['validator_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    result['input_sha256'] = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                              for run in ('sequential', 'dask')
                              for path in sorted((root / run).rglob('*')) if path.is_file()}
    if (root / 'acceptance.json').exists():
        raise FileExistsError('Preserve the previous acceptance report before validating again')
    (root / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Airline: sequential + Dask/search, 15 generations, 300s search budget each; no model tuning')
        return
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    for backend in ('sequential', 'dask'):
        subprocess.run([sys.executable, '-m', 'benchmark.industrial.experiments.evolution_parallelism',
                        '--execute', '--task', 'ts_forecasting', '--backend', backend, '--lifecycle', 'search',
                        '--generations', '15', '--budget-seconds', '300', '--workers', '4',
                        '--output-dir', str(root / backend)], check=True)
    result = validate_pair(root)
    from benchmark.industrial.evaluation.evolution_parallelism import build_search_comparison
    from benchmark.industrial.visualization.evolution_parallelism import render_evolution_parallelism_pack
    render_evolution_parallelism_pack(build_search_comparison([root / 'sequential', root / 'dask']), root / 'report')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit('Paired acceptance failed; inputs preserved without retuning')


if __name__ == '__main__':
    main()

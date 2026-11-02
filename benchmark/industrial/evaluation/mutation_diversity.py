"""Ingest TRAIN-only mutation study traces without model execution."""
from collections import Counter
import json


def jsonl(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()] if path.is_file() else []


def trial_row(directory, job):
    path = directory / 'summary.json'
    if not path.is_file():
        failure = directory / 'failure.json'
        return {**job, 'status': 'failed', 'failure': json.loads(failure.read_text('utf-8')) if failure.is_file() else 'missing summary'}
    s = json.loads(path.read_text('utf-8'))
    if s['test_loss'] is not None or s['source']['official_test_used']:
        raise ValueError('Study consumed official TEST')
    if s['source'].get('outer_used_for_search') is not False:
        raise ValueError('Study must keep the control holdout separate from search')
    prepared = jsonl(directory / 'preparation.jsonl')
    native = [row for p in (directory / 'traces').glob('evaluations_*.jsonl') for row in jsonl(p)]
    mutations = jsonl(directory / 'mutations.jsonl')
    actions = Counter(row['action'] for row in mutations)
    reasons = Counter()
    for row in mutations:
        reasons.update((row.get('planning') or {}).get('reason_counts', {}))
    completed = json.loads((directory / 'completed_evaluations.json').read_text('utf-8'))
    successful = {row['evaluation_context']['configuration_id'] for row in completed}
    all_ids = {row['configuration_id'] for row in native}
    selected = json.loads((directory / 'selected_graph.json').read_text('utf-8'))
    status = s['status'] if job.get('status') == 'finished' and job.get('exit_code') == 0 else job['status']
    return dict(task=job['task'], arm=job['arm'], seed=job['seed'], status=status, exit_code=job.get('exit_code'),
                end_to_end_seconds=job['end_to_end_seconds'], search_seconds=s['search_wall_seconds'],
                generations=s['completed_evolved_generations'],
                evaluations=len(native), successful=len(completed),
                objective_valid_returns=sum(bool(r['valid']) for r in native),
                unique_attempted=len(all_ids), unique_successful=len(successful),
                unique_successful_fraction=len(successful) / len(native) if native else 0,
                raw_unique=len({r['before_id'] for r in prepared}),
                prepared_unique=len({r['configuration_id'] for r in prepared}),
                structures=len({r['structure_id'] for r in prepared}), topologies=len({r['topology_id'] for r in prepared}),
                inner_loss=s['validation_loss'], outer_train_loss=s['outer_train_loss'],
                peak_rss_gib=s['processes']['peak_rss_sum_bytes'] / 1024**3, cpu_seconds=s['processes']['cpu_seconds'],
                failed_evaluations=s['diagnostics'].get('evaluations_failed', 0),
                mutation_statuses=dict(Counter(r['status'] for r in mutations)), actions=dict(actions), planning_reasons=dict(reasons),
                winner_nodes=len(selected['nodes']), train_sha256=s['source']['train_sha256'], outer_sha256=s['source']['outer_sha256'],
                folds=s['source']['folds'], test_used=False, torch_unchanged=s['torch_flags_before'] == s['torch_flags_after'],
                sessions_closed=s['owned_sessions_closed'])

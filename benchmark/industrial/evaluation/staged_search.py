"""Read-only diagnostics for paired TRAIN-only staged searches."""

from collections import Counter
from hashlib import sha256
import json

from .mutation_diversity import jsonl, trial_row


def staged_trial_row(directory, job):
    row = trial_row(directory, {**job, 'arm': job['mode']})
    if not (directory / 'summary.json').exists():
        return row
    summary = json.loads((directory / 'summary.json').read_text('utf-8'))
    completed = json.loads((directory / 'completed_evaluations.json').read_text('utf-8'))
    by_configuration = {entry['evaluation_context']['configuration_id']: entry['evaluation_context']['configuration']
                        for entry in completed}
    prepared = jsonl(directory / 'preparation.jsonl')
    successful = [p for p in prepared if p['configuration_id'] in by_configuration]
    stage_trace = json.loads((directory / 'diagnostics/mutation_schedule.json').read_text('utf-8'))
    diagnostics = summary['dask_diagnostics']
    branch_counts = [max((len(n['parents']) for n in cfg['nodes']), default=0) for cfg in by_configuration.values()]
    operations = Counter(n['operation'] for cfg in by_configuration.values() for n in cfg['nodes'])
    selected = json.loads((directory / 'selected_graph.json').read_text('utf-8'))
    selected_id = 'industrial-evaluation@1:' + sha256(
        json.dumps(selected, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    if selected_id not in by_configuration:
        raise ValueError('Selected graph differs from the completed configuration archive')
    row.update(mode=job['mode'], successful_structures=len({p['structure_id'] for p in successful}),
               successful_topologies=len({p['topology_id'] for p in successful}),
               maximum_successful_arity=max(branch_counts, default=0),
               successful_configs_with_three_or_more_parents=sum(count >= 3 for count in branch_counts),
               operation_counts=dict(operations), stage_events=stage_trace['events'],
               stage_names=[event['name'] for event in stage_trace['events']],
               dask_sessions=summary['dask_sessions'], dask_task_count=sum(d['task_count'] for d in diagnostics),
               dask_compute_seconds=sum(d['compute_seconds'] for d in diagnostics),
               profile_collected=all(d['profile_status'] == 'collected' and not d['errors'] for d in diagnostics),
               phases=summary['phases'], failure_counts=summary['diagnostics']['evaluation_failure_counts'],
               search_budget_overrun_seconds=summary['search_budget_overrun_seconds'],
               minimum_generations_met=summary['minimum_generations_met'], selected_configuration_id=selected_id,
               outer_role=summary['source']['outer_role'])
    return row


def paired_differences(rows):
    """Do not pool losses from unrelated tasks or discard missing/failed pairs."""
    grouped = {}
    for row in rows:
        key = row['task'], row['seed']
        modes = grouped.setdefault(key, {})
        if row['mode'] in modes:
            raise ValueError('Duplicate mode within a task/seed pair')
        modes[row['mode']] = row
    pairs = []
    for (task, seed), modes in sorted(grouped.items()):
        first, second = modes.get('fixed'), modes.get('staged')
        if first is None or second is None or any(r.get('outer_train_loss') is None for r in (first, second)):
            pairs.append({'task': task, 'seed': seed, 'status': 'missing_pair'})
            continue
        if any(r['status'] not in ('succeeded', 'incomplete') or not r['torch_unchanged']
               or not r['sessions_closed'] or r['dask_sessions'] != 1 for r in (first, second)):
            pairs.append({'task': task, 'seed': seed, 'status': 'invalid_pair'})
            continue
        if (first['train_sha256'] != second['train_sha256'] or first['outer_sha256'] != second['outer_sha256']
                or first['folds'] != second['folds']):
            raise ValueError('A paired comparison changed its training data or folds')
        pairs.append({'task': task, 'seed': seed, 'status': 'observed',
                      'fixed_loss': first['outer_train_loss'], 'staged_loss': second['outer_train_loss'],
                      'loss_delta_staged_minus_fixed': second['outer_train_loss'] - first['outer_train_loss'],
                      'unique_successful_delta': second['unique_successful'] - first['unique_successful'],
                      'successful_structures_delta': second['successful_structures'] - first['successful_structures'],
                      'cpu_ratio_staged_over_fixed': second['cpu_seconds'] / first['cpu_seconds'] if first['cpu_seconds'] else None,
                      'both_minimum_generations_met': first['minimum_generations_met'] and second['minimum_generations_met']})
    return pairs

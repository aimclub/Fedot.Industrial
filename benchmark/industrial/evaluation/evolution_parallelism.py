"""Inspectable aggregation for paired search reports; failures stay visible."""
from collections import Counter
from itertools import permutations
import json
from pathlib import Path

import pandas as pd


def read_jsonl(paths):
    return [json.loads(line) for path in paths for line in Path(path).read_text('utf-8').splitlines() if line]


def _work_match(base, row):
    same_data = (row.status == base.status == 'succeeded' and row.implementation is not None
                 and base.implementation is not None
                 and row.train_sha256 == base.train_sha256 and row.native_threads == base.native_threads
                 and row.workload == base.workload and row.implementation == base.implementation)
    same_pool = same_data and row.pool_ids == base.pool_ids
    same_scores = same_pool and row.pool_scores == base.pool_scores
    same_attempts = (same_pool and row.attempt_evidence_complete and base.attempt_evidence_complete
                     and row.attempts == base.attempts and row.failure_types == base.failure_types)
    return same_pool, same_scores, same_attempts


def build_search_comparison(run_dirs):
    run_dirs = tuple(map(Path, run_dirs))
    if len({directory.name.casefold() for directory in run_dirs}) != len(run_dirs):
        raise ValueError('Comparison requires distinct run directory names')
    rows, operations, generations, processes, task_stream, evaluations, profiles = [], [], [], [], [], [], []
    for directory in run_dirs:
        label = directory.name
        summary_path = directory / 'summary.json'
        if not summary_path.is_file():
            failure = json.loads((directory / 'failure.json').read_text('utf-8')
                                 ) if (directory / 'failure.json').is_file() else {}
            rows.append({'run': label, 'status': 'failed', 'failure': failure.get('message', 'Missing summary')})
            continue
        value = json.loads(summary_path.read_text('utf-8'))
        config = value['request']
        if config['pilot']:
            raise ValueError('Pilot data must not be presented as a full search')
        supervisor_path = directory / 'supervisor.json'
        supervisor = json.loads(supervisor_path.read_text('utf-8')) if supervisor_path.is_file() else {}
        row = {'run': label, 'task': config['task'], 'backend': config['backend'],
               'lifecycle': config['lifecycle'], 'protocol': config['protocol'], 'status': value['status'],
               'generations': value['completed_evolved_generations'],
               'minimum_generations_met': value['minimum_generations_met'],
               'wall_seconds': value['wall_seconds'], 'end_to_end_seconds': supervisor.get('end_to_end_seconds'),
               'search_seconds': value['search_seconds'], 'startup_seconds': value['startup_seconds'],
               'search_wall_seconds': value['search_wall_seconds'],
               'search_budget_overrun_seconds': value['search_budget_overrun_seconds'],
               'refit_seconds': value['refit_seconds'], 'predict_seconds': value['predict_seconds'],
               'completed': value['completed_evaluations'], 'unique': value['unique_configurations'],
               'validation_loss': value['validation_loss'], 'test_loss': value['test_loss'],
               'peak_rss_gib': value['processes']['peak_rss_sum_bytes'] / 1024**3,
               'cpu_seconds': value['processes']['cpu_seconds'],
               'train_sha256': value['source']['train_sha256'], 'native_threads': config['native_threads'],
               'dataset': value['source']['dataset'], 'workers': 1 if config['backend'] == 'sequential' else config['workers']}
        row['unique_per_minute'] = 60 * row['unique'] / row['search_wall_seconds']
        diagnostics = value['diagnostics']
        row['failed_evaluations'] = diagnostics.get('evaluations_failed', 0)
        row['repeated_evaluations'] = row['completed'] - row['unique']
        row['failure_types'] = json.dumps(diagnostics.get('evaluation_failure_counts', {}), sort_keys=True)
        row['fitness_reuses'] = diagnostics.get('evaluations_reused', 0)
        row['proposal_rejections'] = json.dumps(diagnostics.get('rejection_counts', {}), sort_keys=True)
        archive = json.loads((directory / 'completed_evaluations.json').read_text('utf-8'))
        ids = Counter(record['evaluation_context']['configuration_id'] for record in archive)
        row['pool_ids'] = json.dumps(ids, sort_keys=True)
        scores = {}
        for record in archive:
            scores.setdefault(record['evaluation_context']['configuration_id'], []).append(record['fitness'][0])
        row['pool_scores'] = json.dumps({key: sorted(values) for key, values in scores.items()}, sort_keys=True)
        row['workload'] = json.dumps({'parameters': value['parameters'], 'folds': value['source']['folds'],
                                      'seed': config['seed']}, sort_keys=True)
        snapshot = directory / 'implementation.json'
        row['implementation'] = (json.dumps(json.loads(snapshot.read_text('utf-8')), sort_keys=True)
                                 if snapshot.is_file() else None)
        rows.append(row)
        for entry in read_jsonl(sorted((directory / 'traces').glob('operations_*.jsonl'))):
            operations.append({'run': label, **entry})
        native_calls = read_jsonl(sorted((directory / 'traces').glob('evaluations_*.jsonl')))
        row['attempt_evidence_complete'] = len(native_calls) == row['completed'] + row['failed_evaluations']
        row['unique_attempted_configurations'] = len({entry['configuration_id'] for entry in native_calls})
        row['successful_repeat_ratio'] = (row['completed'] - row['unique']) / \
            row['completed'] if row['completed'] else None
        row['attempts'] = json.dumps(Counter(json.dumps({
            'id': entry['configuration_id'], 'valid': entry['valid'], 'fitness': entry['fitness']}, sort_keys=True)
            for entry in native_calls), sort_keys=True)
        for entry in native_calls:
            evaluations.append({'run': label, **entry})
        for entry in json.loads((directory / 'generations.json').read_text('utf-8')):
            generations.append({'run': label, **entry})
        sample_path = directory / 'process_samples.csv'
        if sample_path.is_file():
            limits = json.loads((directory / 'limits.json').read_text('utf-8'))
            launched_at = limits['deadline_unix'] - limits['search_budget_seconds']
            for entry in pd.read_csv(sample_path).to_dict('records'):
                processes.append({'run': label, 'launch_elapsed_seconds': entry['timestamp'] - launched_at, **entry})
        for path in sorted((directory / 'dask').glob('*/task_stream.json')):
            for task in json.loads(path.read_text('utf-8')):
                for action in task.get('startstops', ()):
                    task_stream.append({'run': label, 'worker': task['worker'], 'key': task['key'],
                                        'action': action['action'], 'start': action['start'], 'stop': action['stop'],
                                        'seconds': action['stop'] - action['start']})
        from .dask_profiles import read_dask_profile
        for path in sorted((directory / 'dask').glob('*/worker_profile.html')):
            for entry in read_dask_profile(path).to_dict('records'):
                profiles.append({'run': label, 'profile': str(path), **entry})
    comparisons = pd.DataFrame(rows)
    pairs = []
    if 'task' in comparisons:
        for _, group in comparisons.groupby(['task', 'protocol'], dropna=False):
            for base, row in permutations((value for _, value in group.iterrows()), 2):
                same_pool, same_scores, same_attempts = _work_match(base, row)
                matched = same_pool and same_scores and same_attempts
                pairs.append({'task': base.task, 'protocol': base.protocol,
                              'reference': base.run, 'compared': row.run,
                              'reference_status': base.status, 'compared_status': row.status,
                              'same_evaluation_work': matched,
                              'search_speedup': base.search_wall_seconds / row.search_wall_seconds if matched else None})
            baseline = group[group.backend == 'sequential']
            if len(baseline) != 1:
                continue
            base = baseline.iloc[0]
            for index, row in group.iterrows():
                same_pool, same_scores, same_attempts = _work_match(base, row)
                comparisons.loc[index, 'same_candidate_set'] = same_pool
                comparisons.loc[index, 'same_fitness_records'] = same_scores
                comparisons.loc[index, 'same_evaluation_work'] = same_pool and same_attempts
                comparisons.loc[index, 'observed_wall_ratio_vs_sequential'] = base.search_wall_seconds / \
                    row.search_wall_seconds
                comparisons.loc[index, 'same_workload_speedup'] = (base.search_wall_seconds / row.search_wall_seconds
                                                                   if same_pool and same_attempts and same_scores else float('nan'))
    frames = {'comparison': comparisons.drop(columns=['pool_ids', 'pool_scores', 'workload', 'attempts', 'implementation'], errors='ignore'),
              'operations': pd.DataFrame(operations), 'generations': pd.DataFrame(generations),
              'processes': pd.DataFrame(processes), 'task_stream': pd.DataFrame(task_stream),
              'evaluations': pd.DataFrame(evaluations), 'profiles': pd.DataFrame(profiles),
              'paired_work': pd.DataFrame(pairs)}
    frames['execution_diagnostics'] = summarize_execution(frames)
    phase_rows, attempt_rows = [], []
    for directory in run_dirs:
        phase_path = directory / 'phase_timings.json'
        if phase_path.is_file():
            phase_rows.append({'run': directory.name, **json.loads(phase_path.read_text('utf-8'))})
        attempts_path = directory / 'dask_attempts.json'
        if attempts_path.is_file():
            attempt_rows.extend({'run': directory.name, **row}
                                for row in json.loads(attempts_path.read_text('utf-8')))
    frames['phase_timings'] = pd.DataFrame(phase_rows)
    frames['infrastructure_attempts'] = pd.DataFrame(attempt_rows)
    native = frames['evaluations']
    frames['failure_examples'] = (native[~native.valid].drop_duplicates(['run', 'configuration_id'])
                                  if not native.empty else pd.DataFrame())
    return frames


def summarize_execution(frames):
    """Worker occupancy is measured, not inferred from payload size or CPU load."""
    rows = []
    tasks, evaluations = frames['task_stream'], frames['evaluations']
    for run in frames['comparison'].to_dict('records'):
        if 'task' not in run or run.get('status') not in ('succeeded', 'incomplete'):
            continue
        row = {'run': run['run'], 'status': run['status'], 'workers': run['workers'],
               'search_wall_seconds': run['search_wall_seconds']}
        if not evaluations.empty:
            native = evaluations[evaluations.run == run['run']]
            if not native.empty:
                row.update(native_calls=len(native), native_valid=int(native.valid.sum()),
                           native_seconds=float(native.seconds.sum()), native_median_seconds=float(native.seconds.median()),
                           native_p95_seconds=float(native.seconds.quantile(.95)))
        if not tasks.empty:
            compute = tasks[(tasks.run == run['run']) & (tasks.action == 'compute')]
            if not compute.empty:
                events = ([(r.start, 1, r.worker) for r in compute.itertuples()]
                          + [(r.stop, -1, r.worker) for r in compute.itertuples()])
                active, peak, occupied_seconds = Counter(), 0, 0.
                previous = min(compute.start)
                for timestamp, delta, worker in sorted(events):
                    busy = sum(count > 0 for count in active.values())
                    occupied_seconds += busy * (timestamp - previous)
                    active[worker] += delta
                    peak = max(peak, sum(count > 0 for count in active.values()))
                    previous = timestamp
                per_worker = compute.groupby('worker').seconds.sum()
                row.update(dask_compute_seconds=float(compute.seconds.sum()),
                           dask_compute_tasks=len(compute), dask_peak_busy_workers=peak,
                           dask_worker_occupancy=occupied_seconds / (run['workers'] * run['search_wall_seconds']),
                           same_worker_interval_overlap_seconds=float(compute.seconds.sum()) - occupied_seconds,
                           dask_task_median_seconds=float(compute.seconds.median()),
                           dask_task_p95_seconds=float(compute.seconds.quantile(.95)),
                           dask_max_worker_seconds=float(per_worker.max()), dask_min_worker_seconds=float(per_worker.min()))
                if 'native_seconds' in row:
                    # Includes adapter/cleanup/protocol work; not a pure serialization measurement.
                    row['dask_wrapper_seconds'] = row['dask_compute_seconds'] - row['native_seconds']
        rows.append(row)
    return pd.DataFrame(rows)

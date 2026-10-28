"""Pure completeness, non-inferiority and stage-cost acceptance checks."""

from collections import defaultdict
from statistics import mean
import math

from .protocol import experiment_matrix
from .runtime_contract import runtime_matches
from .preparation import archive_evidence_matches


def assess_study(protocol, records):
    config = protocol.to_record()
    criteria = config['acceptance']
    expected = {row['ordinal']: row for row in experiment_matrix(protocol)}
    by_id = {}
    problems = []
    for record in records:
        uid = record.get('ordinal')
        if uid in by_id or uid not in expected:
            problems.append(f'Duplicate or unknown experiment ordinal {uid}')
            continue
        by_id[uid] = record
        if any(record.get(key) != value for key, value in expected[uid].items()):
            problems.append(f'Experiment identity mismatch {uid}')
    missing = sorted(set(expected) - by_id.keys())
    pairs, grouped = [], defaultdict(list)
    for offset in range(0, len(expected), 2):
        first, second = by_id.get(offset), by_id.get(offset + 1)
        if first is None or second is None:
            continue
        if first.get('status') != 'succeeded' or second.get('status') != 'succeeded':
            problems.append(f'Failed pair {offset // 2}')
            continue
        modes = {row['mode']: row for row in (first, second)}
        if set(modes) != {'legacy', 'planned'}:
            problems.append(f'Invalid modes for pair {offset // 2}')
            continue
        old, new = modes['legacy'], modes['planned']
        fields = ('quality_loss', 'baseline_loss', 'search_seconds', 'refit_seconds',
                  'predict_seconds', 'peak_rss_bytes')
        if any(type(row.get(field)) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0
               for row in (old, new) for field in fields):
            problems.append(f'Invalid measurement for pair {offset // 2}')
            continue
        task, dataset = old['scenario']['task'], old['scenario']['name']
        checks = {'matching_data': old['source'] == new['source']}
        for field, prefix in [('search_seconds', 'time'), ('refit_seconds', 'refit'),
                              ('predict_seconds', 'predict'), ('peak_rss_bytes', 'memory')]:
            allowance = criteria[f'{prefix}_allowance_bytes' if prefix == 'memory' else f'{prefix}_allowance_seconds']
            checks[field] = new[field] <= criteria[f'{prefix}_ratio'] * old[field] + allowance
        degradation = new['quality_loss'] - old['quality_loss']
        checks['worst_seed_quality'] = degradation <= criteria['worst_seed_tolerance'][task]
        for row in (old, new):
            mode = row['mode']
            generations = row['generations'][:config['generations']]
            checks[f'{mode}_full_populations'] = (
                len(generations) == config['generations']
                and all(not g.get('is_final', False) and g['population_size'] >= 1
                        and g['valid_fitness_count'] == g['population_size']
                        for g in generations)
                and sum(g['new_successes'] > 0 for g in generations[1:]) >= criteria['minimum_evolved_generations'])
            calls = row['objective_calls']
            succeeded = row['diagnostics']['evaluations_succeeded']
            failures = row['diagnostics']['evaluations_failed']
            attempts = succeeded + failures
            checks[f'{mode}_accounting'] = (
                attempts == len(calls) and succeeded == sum(call['valid'] for call in calls))
            checks[f'{mode}_failure_fraction'] = attempts > 0 and failures / attempts <= (
                criteria['max_failed_evaluation_fraction'])
            checks[f'{mode}_native_folds'] = all(call['fold_count'] == 1 for call in calls if call['valid'])
            checks[f'{mode}_cpu'] = row['torch_threads'] == 1 and not row['cuda_visible'] and all(
                item['num_threads'] == 1 for item in row['threadpools'])
            checks[f'{mode}_frozen_torch'] = runtime_matches(row)
            checks[f'{mode}_completed_archive'] = archive_evidence_matches(row)
        pair = {'task': task, 'dataset': dataset, 'seed': old['seed'], 'degradation': degradation,
                'legacy_loss': old['quality_loss'], 'planned_loss': new['quality_loss'],
                'legacy_baseline_gain': old['baseline_loss'] - old['quality_loss'],
                'planned_baseline_gain': new['baseline_loss'] - new['quality_loss'],
                'legacy_search_seconds': old['search_seconds'], 'planned_search_seconds': new['search_seconds'],
                'checks': checks, 'accepted': all(checks.values())}
        pairs.append(pair)
        grouped[(task, dataset)].append(pair)
    datasets = []
    for (task, dataset), rows in sorted(grouped.items()):
        degradation = mean(row['degradation'] for row in rows)
        datasets.append({'task': task, 'dataset': dataset, 'pairs': len(rows), 'mean_degradation': degradation,
                         'quality_accepted': len(rows) == len(config['seeds'])
                         and degradation <= criteria['quality_tolerance'][task],
                         'legacy_baseline_gain': mean(r['legacy_baseline_gain'] for r in rows),
                         'planned_baseline_gain': mean(r['planned_baseline_gain'] for r in rows)})
    nonvacuity = {task: any(row['task'] == task and min(row['legacy_baseline_gain'], row['planned_baseline_gain'])
                            >= criteria['minimum_baseline_gain'] for row in datasets)
                  for task in config['operations']}
    complete = not missing and not problems and len(pairs) * 2 == len(expected)
    accepted = complete and all(row['accepted'] for row in pairs) and all(
        row['quality_accepted'] for row in datasets) and all(nonvacuity.values())
    return {'schema_version': 'industrial_population_report@1', 'protocol_sha256': protocol.digest,
            'status': 'accepted' if accepted else 'rejected' if complete else 'incomplete',
            'expected_runs': len(expected), 'observed_runs': len(records), 'missing': missing,
            'problems': problems, 'nonvacuity': nonvacuity, 'pairs': pairs, 'datasets': datasets}


def render_report(report):
    lines = ['# Full-Population Mutation Study', '', f"Acceptance: **{report['status']}**", '',
             f"Runs: {report['observed_runs']}/{report['expected_runs']}",
             f"Protocol: `{report['protocol_sha256']}`", '',
             '| Dataset | Task | Pairs | Mean Loss Change | Quality Accepted |',
             '|---|---|---:|---:|---|']
    for row in report['datasets']:
        lines.append(f"| {row['dataset']} | {row['task']} | {row['pairs']} | "
                     f"{row['mean_degradation']:.6f} | {row['quality_accepted']} |")
    lines += ['', '## Paired Checks', '', '| Dataset | Seed | Legacy Search s | Planned Search s | Failed Checks |',
              '|---|---:|---:|---:|---|']
    for row in report['pairs']:
        failures = ', '.join(name for name, passed in row['checks'].items() if not passed) or '-'
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['legacy_search_seconds']:.3f} | "
                     f"{row['planned_search_seconds']:.3f} | {failures} |")
    lines += ['', 'Missing: ' + str(report['missing']), 'Problems: ' + str(report['problems']),
              'Non-vacuity: ' + str(report['nonvacuity']), '',
              'Negative loss change favors planning. This is bounded empirical acceptance, '
              'not proof of universal speedup.',
              'Raw results include stage times, native-inclusive RSS, original predictions, '
              'splits and population evidence.']
    return '\n'.join(lines) + '\n'

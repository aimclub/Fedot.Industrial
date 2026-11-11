import json

import pytest

from benchmark.industrial.evaluation.mutation_diversity import trial_row
from benchmark.industrial.visualization.mutation_diversity import render_mutation_diversity_report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


@pytest.fixture
def trial(tmp_path):
    summary = dict(status='succeeded', test_loss=None, outer_train_loss=.25,
                   validation_loss=.2, search_wall_seconds=300., completed_evolved_generations=15,
                   source=dict(official_test_used=False, outer_used_for_search=False,
                               train_sha256='train', outer_sha256='outer', folds=[{'train_end': 30}]),
                   processes=dict(peak_rss_sum_bytes=2 * 1024**3, cpu_seconds=450.),
                   diagnostics=dict(evaluations_failed=1),
                   torch_flags_before={'threads': 1}, torch_flags_after={'threads': 1},
                   owned_sessions_closed=True)
    write_json(tmp_path / 'summary.json', summary)
    write_json(tmp_path / 'selected_graph.json', {'nodes': [{'operation': 'ridge'}]})
    write_jsonl(tmp_path / 'traces/evaluations_1.jsonl', [
        dict(configuration_id='a', valid=True), dict(configuration_id='a', valid=True),
        dict(configuration_id='b', valid=False), dict(configuration_id='c', valid=True)])
    write_json(tmp_path / 'completed_evaluations.json', [
        {'evaluation_context': {'configuration_id': name}} for name in ('a', 'a', 'c')])
    write_jsonl(tmp_path / 'preparation.jsonl', [
        dict(before_id='a1', configuration_id='a', structure_id='s1', topology_id='t'),
        dict(before_id='a2', configuration_id='a', structure_id='s1', topology_id='t'),
        dict(before_id='c', configuration_id='c', structure_id='s2', topology_id='t')])
    write_jsonl(tmp_path / 'mutations.jsonl', [
        dict(action='single_add', status='accepted', planning={'reason_counts': {'constraint': 100}}),
        dict(action='single_drop', status='rejected', planning=None)])
    job = dict(task='ts_forecasting', arm='baseline', seed=42, status='finished',
               exit_code=0, end_to_end_seconds=303.)
    return tmp_path, summary, job


def test_ingestion_distinguishes_evaluations_configurations_and_normalization(trial):
    root, _, job = trial
    row = trial_row(root, job)
    assert row['evaluations'] == 4 and row['successful'] == 3
    assert row['unique_attempted'] == 3 and row['unique_successful'] == 2
    assert row['unique_successful_fraction'] == .5
    assert (row['raw_unique'], row['prepared_unique'], row['structures'], row['topologies']) == (3, 2, 2, 1)
    assert row['planning_reasons'] == {'constraint': 100}
    assert row['mutation_statuses'] == {'accepted': 1, 'rejected': 1}
    assert row['peak_rss_gib'] == 2 and row['winner_nodes'] == 1
    assert row['torch_unchanged'] and row['sessions_closed'] and not row['test_used']


def test_valid_return_after_deadline_is_not_an_accepted_configuration(trial):
    root, _, job = trial
    trace = root / 'traces/evaluations_1.jsonl'
    with trace.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(configuration_id='late', valid=True)) + '\n')
    row = trial_row(root, job)
    assert row['objective_valid_returns'] == 4
    assert row['successful'] == 3
    assert row['unique_successful'] == 2


@pytest.mark.parametrize('field,value', [('test_loss', .1), ('official_test_used', True)])
def test_ingestion_rejects_official_test_consumption(trial, field, value):
    root, summary, job = trial
    if field == 'test_loss':
        summary[field] = value
    else:
        summary['source'][field] = value
    write_json(root / 'summary.json', summary)
    with pytest.raises(ValueError, match='official TEST'):
        trial_row(root, job)


def test_missing_summary_preserves_failure_details(tmp_path):
    write_json(tmp_path / 'failure.json', {'message': 'worker disconnected'})
    row = trial_row(tmp_path, dict(task='classification', arm='structural', seed=123, status='failed'))
    assert row['status'] == 'failed'
    assert row['failure']['message'] == 'worker disconnected'
    assert 'outer_train_loss' not in row


def test_ingestion_rejects_control_holdout_used_for_search(trial):
    root, summary, job = trial
    summary['source']['outer_used_for_search'] = True
    write_json(root / 'summary.json', summary)
    with pytest.raises(ValueError, match='control holdout'):
        trial_row(root, job)


def test_supervisor_failure_is_not_masked_by_existing_summary(trial):
    root, _, job = trial
    row = trial_row(root, dict(job, status='failed', exit_code=1))
    assert row['status'] == 'failed'
    assert row['exit_code'] == 1


def test_report_preserves_an_all_failed_series(tmp_path):
    rows = [dict(task='classification', arm='baseline', seed=42,
                 status='failed', failure='worker disconnected')]
    render_mutation_diversity_report(rows, tmp_path / 'failed_report')
    saved = json.loads((tmp_path / 'failed_report/results.json').read_text('utf-8'))
    assert saved == rows
    assert 'failed' in (tmp_path / 'failed_report/summary.md').read_text('utf-8')


def test_incomplete_generation_count_is_not_hidden(trial):
    root, summary, job = trial
    summary.update(status='incomplete', completed_evolved_generations=9)
    write_json(root / 'summary.json', summary)
    row = trial_row(root, job)
    assert row['status'] == 'incomplete' and row['generations'] == 9


def test_report_saves_source_table_and_refuses_changed_training_splits(trial, tmp_path):
    root, _, job = trial
    a = trial_row(root, job)
    b = dict(a, arm='structural', unique_successful=3, outer_train_loss=.3)
    report = tmp_path / 'report'
    render_mutation_diversity_report([a, b], report)
    assert json.loads((report / 'results.json').read_text('utf-8')) == [a, b]
    assert (report / 'results.csv').is_file() and (report / 'summary.md').is_file()
    assert (report / 'ts_forecasting_pairs.png').stat().st_size > 1000
    with pytest.raises(FileExistsError):
        render_mutation_diversity_report([a, b], report)
    with pytest.raises(ValueError, match='Training splits changed'):
        render_mutation_diversity_report([a, dict(b, train_sha256='different')], tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()

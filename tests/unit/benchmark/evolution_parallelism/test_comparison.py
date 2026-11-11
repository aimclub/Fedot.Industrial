import json

import pytest

from benchmark.industrial.evaluation.evolution_parallelism import build_search_comparison


def make_run(root, backend, identities=('a', 'b'), scores=(1., 2.), pilot=False, status='succeeded'):
    root.mkdir()
    (root / 'traces').mkdir()
    value = {
        'request': {'task': 'regression', 'backend': backend, 'lifecycle': 'search',
                    'protocol': 'generations', 'pilot': pilot, 'seed': 42, 'workers': 4, 'native_threads': 1},
        'status': status, 'completed_evolved_generations': 15, 'minimum_generations_met': True,
        'wall_seconds': 10, 'search_seconds': 4, 'search_wall_seconds': 8 if backend == 'sequential' else 4,
        'search_budget_overrun_seconds': 0, 'startup_seconds': 1, 'refit_seconds': 1, 'predict_seconds': .1,
        'completed_evaluations': len(identities), 'unique_configurations': len(set(identities)),
        'validation_loss': min(scores), 'test_loss': 3, 'parameters': {'ridge': {'alpha': 1}},
        'processes': {'peak_rss_sum_bytes': 1024**3, 'cpu_seconds': 10},
        'diagnostics': {'evaluations_failed': 2, 'evaluation_failure_counts': {'InvalidFitness': 2}},
        'source': {'train_sha256': 'frozen', 'dataset': 'fixture', 'folds': [{'train': [0], 'validation': [1]}]},
    }
    archive = [{'evaluation_context': {'configuration_id': identity}, 'fitness': [score]}
               for identity, score in zip(identities, scores)]
    for name, body in [('summary', value), ('completed_evaluations', archive), ('generations', [])]:
        (root / f'{name}.json').write_text(json.dumps(body), encoding='utf-8')
    (root / 'implementation.json').write_text(json.dumps({'runtime.py': 'f' * 64}), encoding='utf-8')
    calls = [dict(configuration_id=identity, valid=True, fitness=score, seconds=1)
             for identity, score in zip(identities, scores)]
    calls += [dict(configuration_id=f'bad-{index}', valid=False, fitness=None, seconds=1) for index in range(2)]
    (root / 'traces' / 'evaluations_fixture.jsonl').write_text(
        ''.join(json.dumps(call) + '\n' for call in calls), encoding='utf-8')
    return root


def test_same_workload_requires_multiset_and_same_scores_and_uses_launch_wall(tmp_path):
    a = make_run(tmp_path / 'seq', 'sequential')
    b = make_run(tmp_path / 'dask', 'dask', identities=('b', 'a'), scores=(2., 1.))
    row = build_search_comparison([a, b])['comparison'].iloc[1]
    assert row.same_candidate_set and row.same_fitness_records
    assert row.same_workload_speedup == 2
    assert row.unique_per_minute == 30
    assert row.failed_evaluations == 2
    assert json.loads(row.failure_types) == {'InvalidFitness': 2}


@pytest.mark.parametrize('identities,scores,status', [
    (('a', 'a', 'b'), (1., 1., 2.), 'succeeded'),
    (('a', 'b'), (1., 2.001), 'succeeded'),
    (('a', 'c'), (1., 2.), 'succeeded'),
    (('a', 'b'), (1., 2.), 'incomplete'),
])
def test_different_work_or_results_never_claim_same_workload_speedup(tmp_path, identities, scores, status):
    import math
    a = make_run(tmp_path / 'seq', 'sequential')
    b = make_run(tmp_path / 'dask', 'dask', identities, scores, status=status)
    row = build_search_comparison([a, b])['comparison'].iloc[1]
    assert math.isnan(row.same_workload_speedup)


def test_pilots_cannot_enter_formal_report(tmp_path):
    with pytest.raises(ValueError, match='Pilot'):
        build_search_comparison([make_run(tmp_path / 'pilot', 'sequential', pilot=True)])


def test_failed_run_is_visible_instead_of_dropped(tmp_path):
    root = tmp_path / 'failed'
    root.mkdir()
    (root / 'failure.json').write_text(json.dumps({'message': 'worker died'}), encoding='utf-8')
    row = build_search_comparison([root])['comparison'].iloc[0]
    assert row.status == 'failed' and row.failure == 'worker died'


def test_different_failed_attempts_block_equal_work_claim(tmp_path):
    import math
    a = make_run(tmp_path / 'seq', 'sequential')
    b = make_run(tmp_path / 'dask', 'dask')
    (b / 'traces' / 'evaluations_1.jsonl').write_text(json.dumps({
        'configuration_id': 'failed-extra', 'valid': False, 'fitness': None, 'seconds': 1}) + '\n', encoding='utf-8')
    row = build_search_comparison([a, b])['comparison'].iloc[1]
    assert row.same_candidate_set
    assert not row.same_evaluation_work
    assert math.isnan(row.same_workload_speedup)


def test_dask_occupancy_uses_all_requested_slots_and_counts_overlap():
    import pandas as pd
    from benchmark.industrial.evaluation.evolution_parallelism import summarize_execution
    frames = {
        'comparison': pd.DataFrame([dict(run='r', task='classification', status='succeeded', workers=4, search_wall_seconds=8)]),
        'evaluations': pd.DataFrame(),
        'task_stream': pd.DataFrame([
            dict(run='r', worker='a', action='compute', start=1, stop=5, seconds=4),
            dict(run='r', worker='b', action='compute', start=2, stop=6, seconds=4),
        ]),
    }
    row = summarize_execution(frames).iloc[0]
    assert row.dask_peak_busy_workers == 2
    assert row.dask_worker_occupancy == .25


def test_pairwise_process_dask_match_does_not_require_successful_sequential(tmp_path):
    a = make_run(tmp_path / 'seq', 'sequential', status='incomplete')
    b = make_run(tmp_path / 'proc', 'process')
    c = make_run(tmp_path / 'dask', 'dask')
    frame = build_search_comparison([a, b, c])['paired_work']
    row = frame[(frame.reference == 'proc') & (frame.compared == 'dask')].iloc[0]
    assert row.same_evaluation_work and row.search_speedup == 1
    assert not frame[frame.reference == 'seq'].same_evaluation_work.any()


def test_missing_native_evidence_never_claims_equal_work(tmp_path):
    import math
    a = make_run(tmp_path / 'seq', 'sequential')
    b = make_run(tmp_path / 'dask', 'dask')
    (b / 'traces' / 'evaluations_fixture.jsonl').unlink()
    row = build_search_comparison([a, b])['comparison'].iloc[1]
    assert row.same_candidate_set and not row.attempt_evidence_complete
    assert not row.same_evaluation_work
    assert math.isnan(row.same_workload_speedup)


def test_duplicate_run_labels_cannot_mix_measurements(tmp_path):
    a = make_run(tmp_path / 'same', 'sequential')
    with pytest.raises(ValueError, match='distinct'):
        build_search_comparison([a, a])


def test_absent_source_snapshot_never_claims_equal_work(tmp_path):
    import math
    a = make_run(tmp_path / 'seq', 'sequential')
    b = make_run(tmp_path / 'dask', 'dask')
    (b / 'implementation.json').unlink()
    row = build_search_comparison([a, b])['comparison'].iloc[1]
    assert not row.same_candidate_set
    assert math.isnan(row.same_workload_speedup)


def test_report_figures_keep_source_tables_beside_them(tmp_path):
    from pathlib import Path
    import pandas as pd
    pytest.importorskip('matplotlib')
    from benchmark.industrial.visualization.evolution_parallelism import render_evolution_parallelism_pack
    directories = [make_run(tmp_path / 'seq', 'sequential'), make_run(tmp_path / 'dask', 'dask')]
    for directory in directories:
        (directory / 'traces' / 'operations_1.jsonl').write_text(json.dumps({
            'operation': 'quantile_extractor', 'stage': 'fit', 'exclusive_seconds': 1}) + '\n', encoding='utf-8')
        (directory / 'limits.json').write_text(json.dumps({
            'deadline_unix': 200, 'search_budget_seconds': 100}), encoding='utf-8')
        pd.DataFrame(dict(sample_id=[0, 1], timestamp=[105., 106.], elapsed_seconds=[0., 1.],
                          rss_bytes=[1000, 1100], cpu_delta_seconds=[0., .5], num_threads=[1, 2])).to_csv(
            directory / 'process_samples.csv', index=False)
    target = tmp_path / 'report'
    frames = build_search_comparison(directories)
    assert frames['processes'].launch_elapsed_seconds.tolist() == [5., 6., 5., 6.]
    artifacts = render_evolution_parallelism_pack(frames, target)
    assert artifacts and all(Path(record.path).stat().st_size > 1000 for record in artifacts)
    assert (target / 'comparison.csv').is_file() and (target / 'operation_totals.csv').is_file()
    assert (target / 'paired_work.csv').is_file() and (target / 'summary.md').is_file()
    assert (target / 'resource_samples.csv').is_file()

from copy import deepcopy

import pytest

from benchmark.industrial.evaluation.staged_search import paired_differences


def row(mode, loss):
    return {'task': 'regression', 'seed': 42, 'mode': mode, 'status': 'succeeded',
            'outer_train_loss': loss, 'train_sha256': 'train', 'outer_sha256': 'control', 'folds': [1, 2, 3],
            'torch_unchanged': True, 'sessions_closed': True, 'dask_sessions': 1, 'cpu_seconds': 100.,
            'unique_successful': 10 if mode == 'fixed' else 15, 'successful_structures': 2 if mode == 'fixed' else 4,
            'minimum_generations_met': True}


def test_pair_reports_quality_and_cost_separately_and_preserves_source():
    rows = [row('fixed', 2.), row('staged', 1.5)]
    before = deepcopy(rows)
    result, = paired_differences(rows)
    assert result['loss_delta_staged_minus_fixed'] == -.5
    assert result['unique_successful_delta'] == 5
    assert result['successful_structures_delta'] == 2
    assert result['cpu_ratio_staged_over_fixed'] == 1.
    assert rows == before


def test_missing_and_invalid_pairs_are_not_dropped_or_presented_as_successful():
    assert paired_differences([row('fixed', 2.)])[0]['status'] == 'missing_pair'
    failed = row('staged', 1.)
    failed['sessions_closed'] = False
    assert paired_differences([row('fixed', 2.), failed])[0]['status'] == 'invalid_pair'


def test_pair_rejects_changed_data_or_duplicate_modes():
    changed = row('staged', 1.)
    changed['folds'] = [3, 2, 1]
    with pytest.raises(ValueError, match='training data'):
        paired_differences([row('fixed', 2.), changed])
    with pytest.raises(ValueError, match='Duplicate'):
        paired_differences([row('fixed', 2.), row('fixed', 2.)])


def test_plot_does_not_magnify_numerical_noise_or_change_recorded_losses(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from benchmark.industrial.visualization import benchmark_results
    from benchmark.industrial.visualization.mutation_diversity import render_mutation_diversity_report
    import json

    plots = []

    def capture(figure, *args):
        plots.append(figure.axes[1].get_ylim())
        plt.close(figure)

    monkeypatch.setattr(benchmark_results, '_save_figure', capture)
    rows = [{**row(mode, loss), 'arm': mode, 'unique_successful': 10}
            for mode, loss in [('fixed', 19.295048), ('staged', 19.295040)]]
    render_mutation_diversity_report(rows, tmp_path / 'report', arm_order=('fixed', 'staged'))
    lower, upper = plots[0]
    assert lower < 19.295040 < 19.295048 < upper
    assert upper - lower > .3
    saved = json.loads((tmp_path / 'report/results.json').read_text('utf-8'))
    assert [r['outer_train_loss'] for r in saved] == [19.295048, 19.295040]

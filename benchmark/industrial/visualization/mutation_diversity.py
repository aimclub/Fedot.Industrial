"""Paired plots and adjacent tables for a small TRAIN-only study."""
from pathlib import Path
import json


def render_mutation_diversity_report(rows, output_dir, *, arm_order=('baseline', 'structural'),
                                     title='Mutation profiles: TRAIN-only pilot'):
    import pandas as pd
    from benchmark.industrial.evaluation.markdown import dataframe_to_markdown
    from benchmark.industrial.visualization.benchmark_results import _save_figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    for task, group in frame.groupby('task'):
        complete = group.dropna(subset=['outer_train_loss']) if 'outer_train_loss' in group else group.iloc[:0]
        if complete.empty:
            continue
        if len(set(complete.train_sha256)) != 1 or len(set(complete.outer_sha256)) != 1 or len({json.dumps(f, sort_keys=True) for f in complete.folds}) != 1:
            raise ValueError('Training splits changed across arms or seeds')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'results.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    frame.to_csv(output_dir / 'results.csv', index=False)
    for task, group in frame.groupby('task'):
        complete = group.dropna(subset=['outer_train_loss']) if 'outer_train_loss' in group else group.iloc[:0]
        if complete.empty:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
        for seed, pair in complete.groupby('seed'):
            ordered = pair.set_index('arm').reindex(arm_order)
            axes[0].plot(list(arm_order), ordered.unique_successful, marker='o', label=str(seed))
            axes[1].plot(list(arm_order), ordered.outer_train_loss, marker='o', label=str(seed))
        axes[0].set(ylabel='Unique successful configurations', xlabel='Mutation profile')
        axes[1].set(ylabel='TRAIN control loss (lower is better)', xlabel='Mutation profile')
        lower, upper = complete.outer_train_loss.min(), complete.outer_train_loss.max()
        padding = max((upper - lower) * .1, max(abs(lower), abs(upper), 1.) * .01)
        axes[1].set_ylim(lower - padding, upper + padding)
        axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)
        axes[0].legend(title='Search seed')
        figure.suptitle(task)
        _save_figure(figure, output_dir / f'{task}_pairs', ('png',))
    columns = [c for c in ('task', 'arm', 'seed', 'status', 'generations', 'evaluations',
                           'unique_successful', 'outer_train_loss', 'winner_nodes', 'search_seconds') if c in frame]
    (output_dir / 'summary.md').write_text('# ' + title + '\n\n' + dataframe_to_markdown(frame[columns]) +
                                           '\n\nA/B are different searches under one time limit, not identical-work speed measurements. '
                                           'Outer TRAIN is a control holdout, never official TEST. Three seeds are exploratory evidence only. '
                                           'Incomplete generation counts and failures remain in the table.\n', encoding='utf-8')
    return rows

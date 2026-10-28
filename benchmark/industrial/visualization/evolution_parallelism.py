"""Report figures built from adjacent, exported measurement tables."""
from pathlib import Path

from benchmark.industrial.evaluation.markdown import dataframe_to_markdown
from benchmark.industrial.visualization.benchmark_results import _save_figure


def render_evolution_parallelism_pack(frames, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(root / f'{name}.csv', index=False)
    comparison = frames['comparison']
    columns = [name for name in ('run', 'status', 'generations', 'completed', 'unique', 'search_wall_seconds',
                                 'end_to_end_seconds', 'failed_evaluations',
                                 'unique_per_minute', 'validation_loss', 'test_loss', 'peak_rss_gib',
                                 'same_candidate_set') if name in comparison]
    (root / 'summary.md').write_text('# Evolution execution comparison\n\n' +
                                     dataframe_to_markdown(comparison[columns]) + '\n\n' +
                                     'Raw Dask reports stay beside each source run. RSS is a process sum, not PSS.\n'
                                     'Timing ratios from a single host run are observations, not general speed guarantees.\n', encoding='utf-8')
    plots = []
    phases = frames.get('phase_timings')
    if phases is not None and not phases.empty:
        with (root / 'summary.md').open('a', encoding='utf-8') as stream:
            stream.write('\n## Этапы исполнения\n\n' + dataframe_to_markdown(
                phases.set_index('run').T.rename_axis('phase').reset_index()) + '\n\n'
                'Создание сеанса входит в исполнение пакетов. Вложенные времена не складываются. '
                'Остаток поиска включает диагностику; это не чистое время сериализации.\n')
    if 'task' not in comparison:
        return ()
    for task, group in comparison.dropna(subset=['task']).groupby('task'):
        figure, axes = plt.subplots(1, 2, figsize=(12, max(4, len(group) * .55)), layout='constrained')
        labels = [f'{row.backend}/{row.lifecycle}/{row.protocol}' for row in group.itertuples()]
        axes[0].barh(labels, group.search_wall_seconds, color='#377eb8')
        axes[0].set_xlabel('Search wall time including setup, seconds')
        axes[1].barh(labels, group.unique_per_minute, color='#4daf4a')
        axes[1].set_xlabel('Unique evaluated configurations / minute')
        figure.suptitle(task)
        plots.extend(_save_figure(figure, root / f'{task}_cost', ('png',)))
    generations = frames['generations']
    if not generations.empty:
        for task, selected in comparison.groupby('task'):
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
            for row in selected.itertuples():
                run = row.run
                frame = generations[(generations.run == run) & ~generations.final]
                label = f'{row.backend}/{row.lifecycle}/{row.protocol}'
                axes[0].plot(frame.elapsed_seconds, frame.unique_configurations, label=label)
                axes[1].plot(frame.elapsed_seconds, frame.best_loss, label=label)
            axes[0].set(xlabel='Elapsed seconds', ylabel='Unique evaluated configurations')
            axes[1].set(xlabel='Elapsed seconds', ylabel='Validation loss (lower is better)')
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc='outside lower center', ncols=1, fontsize=8)
            plots.extend(_save_figure(figure, root / f'{task}_progress', ('png',)))
    operations = frames['operations']
    if not operations.empty:
        totals = operations.groupby(['run', 'operation', 'stage']).exclusive_seconds.agg(['sum', 'count']).reset_index()
        totals.to_csv(root / 'operation_totals.csv', index=False)
        expensive = totals.groupby('operation')['sum'].sum().sort_values().tail(12)
        figure, axis = plt.subplots(figsize=(9, max(4, len(expensive) * .45)), layout='constrained')
        axis.barh(expensive.index, expensive.values, color='#e41a1c')
        axis.set_xlabel('Aggregate exclusive operation time, seconds (all runs)')
        plots.extend(_save_figure(figure, root / 'operation_cost', ('png',)))
    processes = frames['processes']
    if not processes.empty:
        samples = processes.groupby(['run', 'sample_id']).agg(
            elapsed_seconds=('launch_elapsed_seconds', 'max'), rss_bytes=('rss_bytes', 'sum'),
            cpu_delta_seconds=('cpu_delta_seconds', 'sum'), threads=('num_threads', 'sum')).reset_index()
        samples.to_csv(root / 'resource_samples.csv', index=False)
        for task, selected in comparison.groupby('task'):
            figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
            for row in selected.itertuples():
                frame = samples[samples.run == row.run].sort_values('sample_id')
                label = f'{row.backend}/{row.lifecycle}/{row.protocol}'
                line, = axes[0].plot(frame.elapsed_seconds, frame.rss_bytes / 1024**3, label=label)
                for axis in axes:
                    axis.axvline(row.search_wall_seconds, color=line.get_color(), linestyle=':', alpha=.35)
                interval = frame.elapsed_seconds.diff().replace(0, float('nan'))
                axes[1].plot(frame.elapsed_seconds, frame.cpu_delta_seconds / interval, label=label)
            axes[0].set(xlabel='Seconds since launch (dotted: search end)', ylabel='Sum of process RSS, GiB')
            axes[1].set(xlabel='Seconds since launch (dotted: search end)', ylabel='CPU seconds / wall second')
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc='outside lower center', ncols=1, fontsize=8)
            plots.extend(_save_figure(figure, root / f'{task}_resources', ('png',)))
    profiles = frames.get('profiles')
    if profiles is not None and not profiles.empty:
        from html import unescape
        from pathlib import PureWindowsPath
        full_search = profiles[profiles.profile.str.replace(
            '\\', '/', regex=False).str.endswith('/search/worker_profile.html')]
        totals = full_search.groupby(['run', 'filename', 'name']).exclusive_sample_share.sum().reset_index()
        totals.to_csv(root / 'profile_hotspots.csv', index=False)
        for run, group in totals.groupby('run'):
            top = group.nlargest(10, 'exclusive_sample_share').sort_values('exclusive_sample_share')
            labels = []
            for row in top.itertuples():
                path = PureWindowsPath(row.filename) if '\\' in row.filename else Path(row.filename)
                labels.append(f'{path.parent.name}/{path.name}: {unescape(row.name)}')
            figure, axis = plt.subplots(figsize=(12, 5.5), layout='constrained')
            axis.barh(labels, top.exclusive_sample_share * 100, color='#984ea3')
            axis.tick_params(axis='y', labelsize=8)
            axis.set_xlabel('Exclusive share of sampled worker stacks, % (not exact CPU time)')
            plots.extend(_save_figure(figure, root / f'{run}_profile_hotspots', ('png',)))
    return tuple(plots)

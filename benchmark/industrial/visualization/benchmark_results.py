from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from benchmark.industrial.core import ArtifactRecord, ensure_directory
from benchmark.industrial.evaluation.markdown import dataframe_to_markdown
from benchmark.industrial.evaluation.result_analysis import (
    ResultAnalysisSpec,
    build_best_per_dataset_frame,
    build_coverage_frame,
    build_dataset_delta_frame,
    build_dataset_difficulty_frame,
    build_generator_usage_frame,
    build_mean_rank_frame,
    build_parameter_metric_frame,
    build_source_delta_frame,
    build_topk_summary_frame,
    normalize_result_table,
)


def render_benchmark_result_analysis_pack(
        frame: pd.DataFrame,
        output_dir: str | Path,
        *,
        spec: ResultAnalysisSpec,
        target_model: str | None = None,
        expected_datasets: Sequence[str] = (),
        expected_dataset_count: int | None = None,
        coverage_unit_column: str = 'dataset_name',
        diagnostics_frame: pd.DataFrame | None = None,
        source_metadata: pd.DataFrame | None = None,
        source_expected_dataset_counts: dict[str, int] | None = None,
        reference_source_labels: Sequence[str] = (),
        target_source_labels: Sequence[str] = (),
        best_target_source_labels: Sequence[str] = (),
        plot_formats: Sequence[str] = ('png',),
) -> tuple[ArtifactRecord, ...]:
    """Render reusable benchmark result tables and plots for notebooks/reports."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target_dir = ensure_directory(output_dir)
    tables_dir = ensure_directory(target_dir / 'tables')
    plots_dir = ensure_directory(target_dir / 'plots')
    manifest: list[ArtifactRecord] = []

    normalized = normalize_result_table(frame, spec)
    best = build_best_per_dataset_frame(normalized, metric_direction=spec.metric_direction)
    mean_rank = build_mean_rank_frame(normalized, metric_direction=spec.metric_direction)
    topk = build_topk_summary_frame(normalized, metric_direction=spec.metric_direction)
    delta = build_dataset_delta_frame(
        normalized,
        target_model=target_model,
        metric_direction=spec.metric_direction,
        reference_source_labels=reference_source_labels,
        target_source_labels=target_source_labels,
    )
    best_target_delta = (
        build_source_delta_frame(
            normalized,
            target_source_labels=best_target_source_labels,
            reference_source_labels=reference_source_labels,
            metric_direction=spec.metric_direction,
            target_strategy='best',
        )
        if best_target_source_labels and reference_source_labels
        else pd.DataFrame()
    )
    difficulty = build_dataset_difficulty_frame(normalized, metric_direction=spec.metric_direction)
    coverage = _build_coverage_rows(
        normalized,
        expected_datasets=expected_datasets,
        expected_dataset_count=expected_dataset_count,
        source_label=spec.source_label,
        unit_column=coverage_unit_column,
        source_expected_dataset_counts=source_expected_dataset_counts or {},
    )
    diagnostics = diagnostics_frame if diagnostics_frame is not None else pd.DataFrame()
    generator_usage = build_generator_usage_frame(diagnostics)
    parameter_metric = build_parameter_metric_frame(normalized, diagnostics)

    for table_name, table in (
            ('normalized_results', normalized),
            ('coverage', coverage),
            ('best_per_dataset', best),
            ('mean_rank', mean_rank),
            ('topk_summary', topk),
            ('dataset_delta', delta),
            ('best_target_delta', best_target_delta),
            ('dataset_difficulty', difficulty),
            ('model_diagnostics', diagnostics),
            ('generator_usage', generator_usage),
            ('parameter_metric_drift', parameter_metric),
    ):
        manifest.extend(_write_table(table, tables_dir / table_name))
    if source_metadata is not None:
        manifest.extend(_write_table(source_metadata, tables_dir / 'source_metadata'))

    summary_path = target_dir / 'summary.md'
    summary_path.write_text(
        _build_summary_markdown(
            spec=spec,
            coverage=coverage,
            mean_rank=mean_rank,
            topk=topk,
            delta=delta,
            best_target_delta=best_target_delta,
            diagnostics=diagnostics,
            source_metadata=source_metadata,
        ),
        encoding='utf-8',
    )
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    if not normalized.empty:
        manifest.extend(_write_table(_build_model_alias_frame(normalized['model_name'].unique()), tables_dir / 'model_aliases'))
        manifest.extend(_render_metric_leaderboard(normalized, plots_dir, spec, plot_formats))
    if not mean_rank.empty:
        manifest.extend(_render_mean_rank(mean_rank, plots_dir, plot_formats))
    if not topk.empty:
        manifest.extend(_render_topk(topk, plots_dir, plot_formats))
    if not delta.empty:
        manifest.extend(_render_delta(delta, plots_dir, plot_formats))

    return tuple(manifest)


def _write_table(frame: pd.DataFrame, path_without_suffix: Path) -> tuple[ArtifactRecord, ...]:
    csv_path = path_without_suffix.with_suffix('.csv')
    md_path = path_without_suffix.with_suffix('.md')
    frame.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(frame, index=False) if not frame.empty else 'No rows.', encoding='utf-8')
    return (
        ArtifactRecord(kind='table', path=str(csv_path), format='csv'),
        ArtifactRecord(kind='summary', path=str(md_path), format='md'),
    )


def _build_summary_markdown(
        *,
        spec: ResultAnalysisSpec,
        coverage: pd.DataFrame,
        mean_rank: pd.DataFrame,
        topk: pd.DataFrame,
        delta: pd.DataFrame,
        best_target_delta: pd.DataFrame,
        diagnostics: pd.DataFrame,
        source_metadata: pd.DataFrame | None,
) -> str:
    lines = [
        f'# Benchmark Result Analysis: {spec.source_label}',
        '',
        f'- Task type: `{spec.task_type or "unknown"}`',
        f'- Metric: `{spec.metric_name}`',
        f'- Metric direction: `{spec.metric_direction}`',
        '',
        '## Sources',
        '',
        dataframe_to_markdown(source_metadata, index=False)
        if source_metadata is not None and not source_metadata.empty
        else 'No source metadata rows.',
        '',
        '## Coverage',
        '',
        dataframe_to_markdown(coverage, index=False) if not coverage.empty else 'No coverage rows.',
        '',
        '## Model Diagnostics',
        '',
        (
            f'- Diagnostic rows: `{len(diagnostics)}`\n'
            f'- Datasets with diagnostics: `{diagnostics["dataset_name"].nunique() if "dataset_name" in diagnostics else 0}`\n'
            f'- Models with diagnostics: `{diagnostics["model_name"].nunique() if "model_name" in diagnostics else 0}`'
        )
        if not diagnostics.empty else 'No model diagnostics rows.',
        '',
        '## Mean Rank',
        '',
        dataframe_to_markdown(mean_rank, index=False) if not mean_rank.empty else 'No rank rows.',
        '',
        '## Top-K Summary',
        '',
        dataframe_to_markdown(topk, index=False) if not topk.empty else 'No top-k rows.',
        '',
        '## Target Delta',
        '',
        dataframe_to_markdown(delta, index=False) if not delta.empty else 'No target delta rows.',
        '',
        '## Best Target Source Delta',
        '',
        dataframe_to_markdown(best_target_delta, index=False)
        if not best_target_delta.empty
        else 'No best target source delta rows.',
    ]
    return '\n'.join(lines)


def _build_coverage_rows(
        normalized: pd.DataFrame,
        *,
        expected_datasets: Sequence[str],
        expected_dataset_count: int | None,
        source_label: str,
        unit_column: str,
        source_expected_dataset_counts: dict[str, int],
) -> pd.DataFrame:
    if normalized.empty or 'source_label' not in normalized.columns:
        return build_coverage_frame(
            normalized,
            expected_datasets=expected_datasets,
            expected_dataset_count=expected_dataset_count,
            source_label=source_label,
            unit_column=unit_column,
        )
    if normalized['source_label'].nunique() <= 1:
        current_source = str(normalized['source_label'].dropna().astype(str).iloc[0])
        return build_coverage_frame(
            normalized,
            expected_datasets=expected_datasets,
            expected_dataset_count=source_expected_dataset_counts.get(current_source, expected_dataset_count),
            source_label=current_source,
            unit_column=unit_column,
        )
    frames = []
    for current_source, group in normalized.groupby(normalized['source_label'].astype(str)):
        frames.append(
            build_coverage_frame(
                group,
                expected_datasets=expected_datasets,
                expected_dataset_count=source_expected_dataset_counts.get(str(current_source), expected_dataset_count),
                source_label=str(current_source),
                unit_column=unit_column,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _render_metric_leaderboard(
        normalized: pd.DataFrame,
        plots_dir: Path,
        spec: ResultAnalysisSpec,
        plot_formats: Sequence[str],
) -> tuple[ArtifactRecord, ...]:
    import matplotlib.pyplot as plt

    metric_means = (
        normalized.groupby('model_name')['metric_value']
        .mean()
        .sort_values(ascending=spec.metric_direction == 'lower')
        .reset_index()
    )
    metric_means['model_alias'] = metric_means['model_name'].map(_short_model_name)
    figure, axis = plt.subplots(figsize=(max(10, 0.55 * len(metric_means)), 5.8))
    axis.bar(metric_means['model_alias'], metric_means['metric_value'])
    axis.set_title(f'Mean {spec.metric_name} by model')
    axis.set_xlabel('Model')
    axis.set_ylabel(spec.metric_name)
    axis.tick_params(axis='x', rotation=35)
    axis.grid(alpha=0.2, axis='y')
    return _save_figure(figure, plots_dir / 'mean_metric_by_model', plot_formats)


def _render_mean_rank(
        mean_rank: pd.DataFrame,
        plots_dir: Path,
        plot_formats: Sequence[str],
) -> tuple[ArtifactRecord, ...]:
    import matplotlib.pyplot as plt

    plot_frame = mean_rank.copy()
    plot_frame['model_alias'] = plot_frame['model_name'].map(_short_model_name)
    figure, axis = plt.subplots(figsize=(max(10, 0.55 * len(plot_frame)), 5.8))
    axis.bar(plot_frame['model_alias'], plot_frame['mean_rank'])
    axis.set_title('Mean rank by model')
    axis.set_xlabel('Model')
    axis.set_ylabel('Mean rank')
    axis.tick_params(axis='x', rotation=35)
    axis.grid(alpha=0.2, axis='y')
    return _save_figure(figure, plots_dir / 'mean_rank_by_model', plot_formats)


def _render_topk(
        topk: pd.DataFrame,
        plots_dir: Path,
        plot_formats: Sequence[str],
) -> tuple[ArtifactRecord, ...]:
    import matplotlib.pyplot as plt

    top_columns = [column for column in topk.columns if column.startswith('top_')]
    if not top_columns:
        return ()
    plot_frame = topk.copy()
    plot_frame['model_alias'] = plot_frame['model_name'].map(_short_model_name)
    plot_frame = plot_frame.set_index('model_alias')[top_columns]
    figure, axis = plt.subplots(figsize=(max(10, 0.55 * len(plot_frame)), 5.8))
    plot_frame.plot(kind='bar', ax=axis)
    axis.set_title('Top-K wins by model')
    axis.set_xlabel('Model')
    axis.set_ylabel('Dataset count')
    axis.tick_params(axis='x', rotation=35)
    axis.grid(alpha=0.2, axis='y')
    return _save_figure(figure, plots_dir / 'topk_wins_by_model', plot_formats)


def _render_delta(
        delta: pd.DataFrame,
        plots_dir: Path,
        plot_formats: Sequence[str],
) -> tuple[ArtifactRecord, ...]:
    import matplotlib.pyplot as plt

    plot_frame = delta.sort_values('improvement', ascending=True)
    figure, axis = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_frame))))
    axis.barh(plot_frame['dataset_name'], plot_frame['improvement'])
    axis.axvline(0.0, color='black', linestyle='--', linewidth=1.0)
    axis.set_title('Target model improvement over best reference')
    axis.set_xlabel('Improvement')
    axis.set_ylabel('Dataset')
    axis.grid(alpha=0.2, axis='x')
    return _save_figure(figure, plots_dir / 'target_delta_by_dataset', plot_formats)


def _build_model_alias_frame(model_names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {'model_name': str(model_name), 'model_alias': _short_model_name(str(model_name))}
            for model_name in sorted({str(name) for name in model_names})
        ]
    )


def _short_model_name(model_name: str) -> str:
    replacements = {
        'KernelEnsembleClassifier_adaptive_all_non_topological': 'KLC adaptive',
        'KernelEnsembleClassifier_shapelet_motif_rbf': 'KLC shapelet',
        'KernelEnsembleClassifier_embedding_nystrom': 'KLC nystrom',
        'KernelEnsembleClassifier_score_baseline_summary': 'KLC baseline',
        'KernelEnsembleRegressor_adaptive_rbf_summary': 'KLR adaptive',
        'KernelEnsembleRegressor_shapelet_rbf': 'KLR shapelet',
        'KernelEnsembleRegressor_embedding_nystrom': 'KLR nystrom',
        'KernelEnsembleRegressor_score_linear_summary': 'KLR linear',
        'KernelEnsembleForecaster_identity_shapelet': 'KLF shapelet',
        'KernelEnsembleForecaster_embedding_nystrom_okhs': 'KLF okhs',
        'LaggedRidgeForecaster': 'Lagged Ridge',
        'NaiveLastValue': 'Naive',
        'Fedot_Industrial_legacy_baseline_features': 'FI legacy features',
        'Fedot_Industrial_legacy_baseline': 'FI legacy base',
        'Fedot_Industrial_legacy_advanced_features': 'FI legacy advanced',
    }
    if model_name in replacements:
        return replacements[model_name]
    shortened = model_name
    for prefix, replacement in (
            ('KernelEnsembleClassifier_', 'KLC '),
            ('KernelEnsembleRegressor_', 'KLR '),
            ('KernelEnsembleForecaster_', 'KLF '),
            ('Fedot_Industrial_', 'FI '),
    ):
        shortened = shortened.replace(prefix, replacement)
    shortened = shortened.replace('_', ' ')
    return shortened[:36]


def _save_figure(
        figure,
        path_without_suffix: Path,
        plot_formats: Sequence[str],
) -> tuple[ArtifactRecord, ...]:
    import matplotlib.pyplot as plt

    manifest: list[ArtifactRecord] = []
    for extension in plot_formats:
        path = path_without_suffix.with_suffix(f'.{extension}')
        figure.savefig(path, dpi=200, bbox_inches='tight')
        manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
    plt.close(figure)
    return tuple(manifest)


__all__ = ['render_benchmark_result_analysis_pack']

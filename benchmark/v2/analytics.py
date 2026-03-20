from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .core import (
    ArtifactRecord,
    ForecastingBenchmarkResult,
    MetricRecord,
    PredictionRecord,
    RunStatus,
    ensure_directory,
    to_plain_data,
    write_json,
)


@dataclass(frozen=True)
class SeriesComparisonResult:
    series_id: str
    dataset_name: str
    model_names: tuple[str, ...]
    metrics_table: pd.DataFrame
    prediction_table: pd.DataFrame
    artifact_manifest: tuple[ArtifactRecord, ...] = ()


def predictions_to_frame(records: tuple[PredictionRecord, ...]) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in records])


def metrics_to_frame(records: tuple[MetricRecord, ...]) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in records])


def runs_to_frame(result: ForecastingBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        row = {
            'run_id': record.run_id,
            'benchmark': record.benchmark,
            'dataset_name': record.dataset_name,
            'subset': record.subset,
            'series_id': record.series_id,
            'model_name': record.model_name,
            'status': record.status.value,
            'message': record.message,
        }
        row.update(record.metrics_summary)
        rows.append(row)
    return pd.DataFrame(rows)


def build_benchmark_leaderboard(
        result: ForecastingBenchmarkResult,
        primary_metric: str | None = None,
) -> pd.DataFrame:
    metric_name = primary_metric or result.aggregate_report.primary_metric
    run_frame = runs_to_frame(result)
    if run_frame.empty:
        return pd.DataFrame(columns=['benchmark', 'dataset_name', 'model_name', metric_name, 'n_series', 'rank'])
    successful = run_frame[run_frame['status'] == RunStatus.SUCCESS.value]
    if successful.empty:
        return pd.DataFrame(columns=['benchmark', 'dataset_name', 'model_name', metric_name, 'n_series', 'rank'])
    leaderboard = (
        successful.groupby(['benchmark', 'dataset_name', 'model_name'])[metric_name]
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': metric_name, 'count': 'n_series'})
        .sort_values(metric_name)
        .reset_index(drop=True)
    )
    leaderboard['rank'] = leaderboard[metric_name].rank(method='dense')
    return leaderboard


def _stable_write_table(frame: pd.DataFrame, path_without_suffix: Path) -> list[ArtifactRecord]:
    artifacts: list[ArtifactRecord] = []
    csv_path = path_without_suffix.with_suffix('.csv')
    frame.to_csv(csv_path, index=False)
    artifacts.append(ArtifactRecord(kind='table', path=str(csv_path), format='csv'))

    tex_path = path_without_suffix.with_suffix('.tex')
    tex_path.write_text(frame.to_latex(index=False, float_format=lambda value: f'{value:.4f}'), encoding='utf-8')
    artifacts.append(ArtifactRecord(kind='table', path=str(tex_path), format='tex'))

    parquet_path = path_without_suffix.with_suffix('.parquet')
    try:
        frame.to_parquet(parquet_path, index=False)
        artifacts.append(ArtifactRecord(kind='structured', path=str(parquet_path), format='parquet'))
    except Exception:
        pass
    return artifacts


def compare_models_on_series(
        result: ForecastingBenchmarkResult,
        series_id: str,
        output_dir: str | Path | None = None,
) -> SeriesComparisonResult:
    predictions = predictions_to_frame(result.prediction_records)
    metrics = metrics_to_frame(result.metric_records)

    series_predictions = predictions[predictions['series_id'] == series_id].copy()
    if series_predictions.empty:
        raise ValueError(f'No prediction records found for series_id={series_id}.')

    dataset_name = str(series_predictions['dataset_name'].iloc[0])
    series_metrics = metrics[(metrics['series_id'] == series_id) & (metrics['horizon_index'].isna())].copy()
    series_metrics = series_metrics.sort_values(['metric_name', 'metric_value', 'model_name'])
    series_predictions = series_predictions.sort_values(['model_name', 'horizon_index'])

    artifact_manifest: list[ArtifactRecord] = []
    if output_dir is not None:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        target_dir = ensure_directory(output_dir)
        pivot = series_predictions.pivot(index='horizon_index', columns='model_name', values='y_pred')
        truth = (
            series_predictions[['horizon_index', 'y_true']]
            .drop_duplicates()
            .sort_values('horizon_index')
            .set_index('horizon_index')
        )

        overlay_figure, overlay_axis = plt.subplots(figsize=(9, 5))
        overlay_axis.plot(truth.index, truth['y_true'], label='truth', linewidth=2.5, color='black')
        for model_name in pivot.columns:
            overlay_axis.plot(pivot.index, pivot[model_name], label=model_name, linewidth=1.8)
        overlay_axis.set_title(f'Forecast Comparison for {series_id}')
        overlay_axis.set_xlabel('Horizon')
        overlay_axis.set_ylabel('Value')
        overlay_axis.legend(frameon=False)
        overlay_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_overlay.{extension}'
            overlay_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(overlay_figure)

        residual_figure, residual_axis = plt.subplots(figsize=(9, 5))
        for model_name in pivot.columns:
            residual_axis.plot(
                pivot.index,
                truth['y_true'].to_numpy() - pivot[model_name].to_numpy(),
                label=model_name,
                linewidth=1.6,
            )
        residual_axis.axhline(0.0, color='black', linestyle='--', linewidth=1)
        residual_axis.set_title(f'Residuals for {series_id}')
        residual_axis.set_xlabel('Horizon')
        residual_axis.set_ylabel('Residual')
        residual_axis.legend(frameon=False)
        residual_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_residuals.{extension}'
            residual_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(residual_figure)

        horizon_metrics = metrics[
            (metrics['series_id'] == series_id)
            & (metrics['metric_name'] == result.aggregate_report.primary_metric)
            & (metrics['horizon_index'].notna())
            ].copy()
        if not horizon_metrics.empty:
            horizon_figure, horizon_axis = plt.subplots(figsize=(9, 5))
            for model_name, group in horizon_metrics.groupby('model_name'):
                horizon_axis.plot(group['horizon_index'], group['metric_value'], label=model_name, linewidth=1.8)
            horizon_axis.set_title(f'Horizon Error Profile for {series_id}')
            horizon_axis.set_xlabel('Horizon')
            horizon_axis.set_ylabel(result.aggregate_report.primary_metric.upper())
            horizon_axis.legend(frameon=False)
            horizon_axis.grid(alpha=0.2)
            for extension in ('png', 'svg'):
                path = target_dir / f'{series_id}_horizon_profile.{extension}'
                horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
                artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
            plt.close(horizon_figure)

    return SeriesComparisonResult(
        series_id=series_id,
        dataset_name=dataset_name,
        model_names=tuple(sorted(series_predictions['model_name'].unique())),
        metrics_table=series_metrics.reset_index(drop=True),
        prediction_table=series_predictions.reset_index(drop=True),
        artifact_manifest=tuple(artifact_manifest),
    )


def render_publication_pack(
        result: ForecastingBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    target_dir = ensure_directory(output_dir or Path(result.config.artifact_spec.output_dir) / result.run_id)
    aggregate_dir = ensure_directory(target_dir / 'aggregate')
    series_dir = ensure_directory(target_dir / 'series')

    manifest: list[ArtifactRecord] = []
    metrics_frame = metrics_to_frame(result.metric_records)
    predictions_frame = predictions_to_frame(result.prediction_records)
    runs_frame = runs_to_frame(result)
    leaderboard = build_benchmark_leaderboard(result)

    for base_name, frame in (
            ('metrics', metrics_frame),
            ('predictions', predictions_frame),
            ('runs', runs_frame),
            ('leaderboard', leaderboard),
    ):
        manifest.extend(_stable_write_table(frame, aggregate_dir / base_name))

    metadata_path = aggregate_dir / 'run_metadata.json'
    metadata_payload = {
        'run_id': result.run_id,
        'task_type': result.config.task_type.value,
        'primary_metric': result.aggregate_report.primary_metric,
        'status_counts': result.aggregate_report.status_counts,
        'dataset_specs': [to_plain_data(spec) for spec in result.config.datasets],
        'model_specs': [to_plain_data(spec) for spec in result.config.models],
    }
    write_json(metadata_path, metadata_payload)
    manifest.append(ArtifactRecord(kind='structured', path=str(metadata_path), format='json'))

    summary_path = aggregate_dir / 'summary.md'
    summary_lines = [
        f'# Forecasting Benchmark Summary: {result.run_id}',
        '',
        f'- Primary metric: `{result.aggregate_report.primary_metric}`',
        f'- Successful runs: `{result.aggregate_report.status_counts.get("success", 0)}`',
        f'- Failed runs: `{result.aggregate_report.status_counts.get("failed", 0)}`',
        f'- Skipped runs: `{result.aggregate_report.status_counts.get("skipped", 0)}`',
        f'- Not available runs: `{result.aggregate_report.status_counts.get("not_available", 0)}`',
        '',
        '## Leaderboard',
        '',
    ]
    if leaderboard.empty:
        summary_lines.append('No successful benchmark runs were recorded.')
    else:
        summary_lines.append(leaderboard.to_markdown(index=False))
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    successful_runs = runs_frame[runs_frame['status'] == RunStatus.SUCCESS.value].copy()
    if not successful_runs.empty:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        primary_metric = result.aggregate_report.primary_metric
        boxplot_figure, boxplot_axis = plt.subplots(figsize=(10, 5))
        successful_runs.boxplot(column=primary_metric, by='model_name', ax=boxplot_axis)
        boxplot_axis.set_title(f'{primary_metric.upper()} Distribution by Model')
        boxplot_axis.set_xlabel('Model')
        boxplot_axis.set_ylabel(primary_metric.upper())
        boxplot_axis.grid(alpha=0.2)
        boxplot_axis.figure.suptitle('')
        for extension in ('png', 'svg'):
            path = aggregate_dir / f'{primary_metric}_distribution.{extension}'
            boxplot_figure.savefig(path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(boxplot_figure)

        horizon_metrics = metrics_frame[
            (metrics_frame['metric_name'] == primary_metric) & (metrics_frame['horizon_index'].notna())
            ].copy()
        if not horizon_metrics.empty:
            horizon_plot = (
                horizon_metrics.groupby(['model_name', 'horizon_index'])['metric_value']
                .mean()
                .reset_index()
            )
            horizon_figure, horizon_axis = plt.subplots(figsize=(10, 5))
            for model_name, group in horizon_plot.groupby('model_name'):
                horizon_axis.plot(group['horizon_index'], group['metric_value'], label=model_name, linewidth=1.8)
            horizon_axis.set_title(f'Horizon vs {primary_metric.upper()}')
            horizon_axis.set_xlabel('Horizon')
            horizon_axis.set_ylabel(primary_metric.upper())
            horizon_axis.legend(frameon=False)
            horizon_axis.grid(alpha=0.2)
            for extension in ('png', 'svg'):
                path = aggregate_dir / f'horizon_vs_{primary_metric}.{extension}'
                horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
                manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
            plt.close(horizon_figure)

        okhs_rows = successful_runs[successful_runs['model_name'].str.contains('okhs', case=False, regex=False)]
        if not okhs_rows.empty:
            baseline_rows = successful_runs[
                ~successful_runs['model_name'].str.contains('okhs', case=False, regex=False)]
            pairwise_rows = []
            for _, okhs_row in okhs_rows.iterrows():
                comparable = baseline_rows[
                    (baseline_rows['benchmark'] == okhs_row['benchmark'])
                    & (baseline_rows['dataset_name'] == okhs_row['dataset_name'])
                    & (baseline_rows['series_id'] == okhs_row['series_id'])
                    ]
                for _, baseline_row in comparable.iterrows():
                    pairwise_rows.append(
                        {
                            'benchmark': okhs_row['benchmark'],
                            'dataset_name': okhs_row['dataset_name'],
                            'series_id': okhs_row['series_id'],
                            'okhs_model': okhs_row['model_name'],
                            'baseline_model': baseline_row['model_name'],
                            primary_metric: okhs_row[primary_metric],
                            f'baseline_{primary_metric}': baseline_row[primary_metric],
                            'delta': okhs_row[primary_metric] - baseline_row[primary_metric],
                        }
                    )
            if pairwise_rows:
                pairwise_frame = pd.DataFrame(pairwise_rows).sort_values('delta')
                manifest.extend(_stable_write_table(pairwise_frame, aggregate_dir / 'okhs_pairwise_comparison'))

    available_series = predictions_frame['series_id'].drop_duplicates().tolist()
    for series_id in available_series[: min(3, len(available_series))]:
        comparison = compare_models_on_series(result, series_id=series_id, output_dir=series_dir / series_id)
        manifest.extend(comparison.artifact_manifest)

    return tuple(manifest)

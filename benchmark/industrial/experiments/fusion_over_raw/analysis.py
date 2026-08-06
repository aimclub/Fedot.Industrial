"""Result analysis helpers for the fusion_over_raw experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from benchmark.industrial.core import ArtifactRecord, ensure_directory
from benchmark.industrial.evaluation.markdown import dataframe_to_markdown
from benchmark.industrial.evaluation.result_analysis import (
    ResultAnalysisSpec,
    build_dataset_delta_frame,
    infer_metric_direction,
)
from benchmark.industrial.experiments.fusion_over_raw.expand import model_family_name
from benchmark.industrial.visualization.benchmark_results import render_benchmark_result_analysis_pack

MINIROCKET_FAMILY = 'MiniRocketRidge'
SINGLE_VIEW_FAMILIES = frozenset({'raw', 'raw_larger', 'stats', 'gaf', 'stft'})


def average_metrics_by_family(
        normalized: pd.DataFrame,
        *,
        metric_name: str = 'f1_macro',
) -> pd.DataFrame:
    """Average seed-suffixed model runs into family-level scores."""
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'model_name',
                'metric_name',
                'metric_value',
                'source_label',
                'task_type',
                'metric_direction',
                'seed_count',
            ]
        )

    frame = normalized.copy()
    if metric_name and 'metric_name' in frame.columns:
        frame = frame[frame['metric_name'].astype(str) == str(metric_name)].copy()
    if frame.empty:
        return average_metrics_by_family(pd.DataFrame(), metric_name=metric_name)

    frame['model_family'] = frame['model_name'].map(model_family_name)
    grouped = (
        frame.groupby(['dataset_name', 'model_family', 'metric_name'], as_index=False)
        .agg(
            metric_value=('metric_value', 'mean'),
            seed_count=('metric_value', 'size'),
            source_label=('source_label', 'first'),
            task_type=('task_type', 'first'),
            metric_direction=('metric_direction', 'first'),
        )
        .rename(columns={'model_family': 'model_name'})
    )
    return grouped.sort_values(['dataset_name', 'model_name']).reset_index(drop=True)


def classify_model_role(model_name: str) -> str:
    family = model_family_name(model_name)
    if family == MINIROCKET_FAMILY or family.startswith('MiniRocketRidge'):
        return 'external'
    if family in SINGLE_VIEW_FAMILIES:
        return 'baseline'
    if '__' in family:
        return 'fusion'
    return 'other'


def build_family_vs_best_single_frame(
        normalized: pd.DataFrame,
        *,
        metric_name: str = 'f1_macro',
        metric_direction: str | None = None,
) -> pd.DataFrame:
    """Compare each fusion family against the best single-view baseline per dataset."""
    family = average_metrics_by_family(normalized, metric_name=metric_name)
    if family.empty:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'fusion_family',
                'fusion_metric',
                'best_single_model',
                'best_single_metric',
                'improvement',
                'relative_improvement_pct',
            ]
        )

    direction = metric_direction or str(family['metric_direction'].dropna().iloc[0])
    higher = direction == 'higher'
    rows = []
    for dataset_name, group in family.groupby('dataset_name'):
        singles = group[group['model_name'].map(classify_model_role) == 'baseline']
        fusions = group[group['model_name'].map(classify_model_role) == 'fusion']
        if singles.empty or fusions.empty:
            continue
        best_idx = singles['metric_value'].idxmax() if higher else singles['metric_value'].idxmin()
        best_single_model = str(singles.loc[best_idx, 'model_name'])
        best_single_metric = float(singles.loc[best_idx, 'metric_value'])
        for _, fusion_row in fusions.iterrows():
            fusion_metric = float(fusion_row['metric_value'])
            improvement = fusion_metric - best_single_metric if higher else best_single_metric - fusion_metric
            denominator = abs(best_single_metric) if best_single_metric != 0 else 1.0
            rows.append(
                {
                    'dataset_name': dataset_name,
                    'fusion_family': str(fusion_row['model_name']),
                    'fusion_metric': fusion_metric,
                    'best_single_model': best_single_model,
                    'best_single_metric': best_single_metric,
                    'improvement': improvement,
                    'relative_improvement_pct': 100.0 * improvement / denominator,
                }
            )
    if not rows:
        return build_family_vs_best_single_frame(pd.DataFrame(), metric_name=metric_name)
    return pd.DataFrame(rows).sort_values(
        ['dataset_name', 'improvement'],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_top_family_by_dataset_frame(
        normalized: pd.DataFrame,
        *,
        top_k: int = 3,
        metric_name: str = 'f1_macro',
        metric_direction: str | None = None,
        roles: Sequence[str] = ('fusion',),
) -> pd.DataFrame:
    family = average_metrics_by_family(normalized, metric_name=metric_name)
    if family.empty:
        return pd.DataFrame(
            columns=['dataset_name', 'rank', 'model_name', 'metric_name', 'metric_value', 'role']
        )

    direction = metric_direction or str(family['metric_direction'].dropna().iloc[0])
    ascending = direction == 'lower'
    role_set = {str(role) for role in roles}
    filtered = family[family['model_name'].map(classify_model_role).isin(role_set)].copy()
    if filtered.empty:
        return build_top_family_by_dataset_frame(pd.DataFrame(), top_k=top_k, metric_name=metric_name)

    filtered['role'] = filtered['model_name'].map(classify_model_role)
    ordered = filtered.sort_values(['dataset_name', 'metric_value'], ascending=[True, ascending])
    ranked = ordered.groupby('dataset_name', as_index=False).head(int(top_k)).copy()
    ranked['rank'] = ranked.groupby('dataset_name').cumcount() + 1
    return ranked[
        ['dataset_name', 'rank', 'model_name', 'metric_name', 'metric_value', 'role']
    ].reset_index(drop=True)


def build_feature_combination_sensitivity_frame(
        normalized: pd.DataFrame,
        *,
        metric_name: str = 'f1_macro',
        metric_direction: str | None = None,
) -> pd.DataFrame:
    """Aggregate fusion families by modality combination and fusion method."""
    family = average_metrics_by_family(normalized, metric_name=metric_name)
    if family.empty:
        return pd.DataFrame(
            columns=[
                'modalities',
                'fusion_method',
                'dataset_count',
                'mean_metric',
                'best_dataset',
                'best_metric',
            ]
        )

    fusions = family[family['model_name'].map(classify_model_role) == 'fusion'].copy()
    if fusions.empty:
        return build_feature_combination_sensitivity_frame(pd.DataFrame(), metric_name=metric_name)

    direction = metric_direction or str(fusions['metric_direction'].dropna().iloc[0])
    higher = direction == 'higher'
    parts = fusions['model_name'].astype(str).str.rsplit('__', n=1)
    fusions['modalities'] = parts.str[0]
    fusions['fusion_method'] = parts.str[1]
    rows = []
    for (modalities, fusion_method), group in fusions.groupby(['modalities', 'fusion_method']):
        best_idx = group['metric_value'].idxmax() if higher else group['metric_value'].idxmin()
        rows.append(
            {
                'modalities': modalities,
                'fusion_method': fusion_method,
                'dataset_count': int(group['dataset_name'].nunique()),
                'mean_metric': float(group['metric_value'].mean()),
                'best_dataset': str(group.loc[best_idx, 'dataset_name']),
                'best_metric': float(group.loc[best_idx, 'metric_value']),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ['mean_metric', 'modalities', 'fusion_method'],
        ascending=[not higher, True, True],
    ).reset_index(drop=True)


def build_minirocket_comparison_frame(
        normalized: pd.DataFrame,
        *,
        metric_name: str = 'f1_macro',
        metric_direction: str | None = None,
        target_strategy: str = 'best_fusion',
) -> pd.DataFrame:
    """Compare MiniRocketRidge against best fusion or a fixed family target."""
    family = average_metrics_by_family(normalized, metric_name=metric_name)
    if family.empty:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'target_model',
                'target_metric',
                'minirocket_metric',
                'improvement_vs_minirocket',
                'relative_improvement_pct',
            ]
        )

    direction = metric_direction or str(family['metric_direction'].dropna().iloc[0])
    higher = direction == 'higher'
    rows = []
    for dataset_name, group in family.groupby('dataset_name'):
        rocket = group[group['model_name'].map(classify_model_role) == 'external']
        if rocket.empty:
            continue
        minirocket_metric = float(rocket['metric_value'].mean())
        if target_strategy == 'best_fusion':
            fusions = group[group['model_name'].map(classify_model_role) == 'fusion']
            if fusions.empty:
                continue
            best_idx = fusions['metric_value'].idxmax() if higher else fusions['metric_value'].idxmin()
            target_model = str(fusions.loc[best_idx, 'model_name'])
            target_metric = float(fusions.loc[best_idx, 'metric_value'])
        elif target_strategy == 'best_single':
            singles = group[group['model_name'].map(classify_model_role) == 'baseline']
            if singles.empty:
                continue
            best_idx = singles['metric_value'].idxmax() if higher else singles['metric_value'].idxmin()
            target_model = str(singles.loc[best_idx, 'model_name'])
            target_metric = float(singles.loc[best_idx, 'metric_value'])
        else:
            raise ValueError(f'Unsupported MiniRocket comparison strategy: {target_strategy}')

        improvement = target_metric - minirocket_metric if higher else minirocket_metric - target_metric
        denominator = abs(minirocket_metric) if minirocket_metric != 0 else 1.0
        rows.append(
            {
                'dataset_name': dataset_name,
                'target_model': target_model,
                'target_metric': target_metric,
                'minirocket_metric': minirocket_metric,
                'improvement_vs_minirocket': improvement,
                'relative_improvement_pct': 100.0 * improvement / denominator,
            }
        )
    if not rows:
        return build_minirocket_comparison_frame(pd.DataFrame(), metric_name=metric_name)
    return pd.DataFrame(rows).sort_values('improvement_vs_minirocket', ascending=False).reset_index(drop=True)


def build_bottleneck_variants_comparison_frame(
        normalized: pd.DataFrame,
        *,
        metric_name: str = 'f1_macro',
        metric_direction: str | None = None,
) -> pd.DataFrame:
    """Placeholder comparison for MMT-104 bottleneck variants.

    Returns an empty frame with a stable schema until bottleneck runs are enabled.
    """
    family = average_metrics_by_family(normalized, metric_name=metric_name)
    if family.empty:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'bottleneck_family',
                'metric_value',
                'best_mvp_family',
                'best_mvp_metric',
                'improvement_vs_mvp',
            ]
        )

    direction = metric_direction or infer_metric_direction(metric_name)
    higher = direction == 'higher'
    bottleneck = family[family['model_name'].astype(str).str.contains('bottleneck', regex=False)].copy()
    mvp = family[
        (family['model_name'].map(classify_model_role) == 'fusion')
        & (~family['model_name'].astype(str).str.contains('bottleneck', regex=False))
    ].copy()
    if bottleneck.empty or mvp.empty:
        return build_bottleneck_variants_comparison_frame(pd.DataFrame(), metric_name=metric_name)

    rows = []
    for dataset_name, group in bottleneck.groupby('dataset_name'):
        mvp_group = mvp[mvp['dataset_name'] == dataset_name]
        if mvp_group.empty:
            continue
        best_idx = mvp_group['metric_value'].idxmax() if higher else mvp_group['metric_value'].idxmin()
        best_mvp_family = str(mvp_group.loc[best_idx, 'model_name'])
        best_mvp_metric = float(mvp_group.loc[best_idx, 'metric_value'])
        for _, row in group.iterrows():
            metric_value = float(row['metric_value'])
            improvement = metric_value - best_mvp_metric if higher else best_mvp_metric - metric_value
            rows.append(
                {
                    'dataset_name': dataset_name,
                    'bottleneck_family': str(row['model_name']),
                    'metric_value': metric_value,
                    'best_mvp_family': best_mvp_family,
                    'best_mvp_metric': best_mvp_metric,
                    'improvement_vs_mvp': improvement,
                }
            )
    if not rows:
        return build_bottleneck_variants_comparison_frame(pd.DataFrame(), metric_name=metric_name)
    return pd.DataFrame(rows).sort_values(
        ['dataset_name', 'improvement_vs_mvp'],
        ascending=[True, False],
    ).reset_index(drop=True)


def render_fusion_over_raw_analysis_pack(
        normalized: pd.DataFrame,
        output_dir: str | Path,
        *,
        metric_name: str = 'f1_macro',
        expected_datasets: Sequence[str] = (),
) -> tuple[ArtifactRecord, ...]:
    """Write shared result_analysis pack plus fusion-specific tables."""
    target_dir = ensure_directory(output_dir)
    tables_dir = ensure_directory(target_dir / 'tables')
    direction = infer_metric_direction(metric_name)
    spec = ResultAnalysisSpec(
        metric_name=metric_name,
        metric_direction=direction,
        source_label='fusion_over_raw',
        task_type='ts_classification',
    )
    family = average_metrics_by_family(normalized, metric_name=metric_name)
    family_vs_best = build_family_vs_best_single_frame(family, metric_name=metric_name, metric_direction=direction)
    top3 = build_top_family_by_dataset_frame(family, top_k=3, metric_name=metric_name, metric_direction=direction)
    sensitivity = build_feature_combination_sensitivity_frame(
        family,
        metric_name=metric_name,
        metric_direction=direction,
    )
    minirocket = build_minirocket_comparison_frame(family, metric_name=metric_name, metric_direction=direction)
    bottleneck = build_bottleneck_variants_comparison_frame(
        family,
        metric_name=metric_name,
        metric_direction=direction,
    )

    # Reuse generic pack on family-averaged frame for ranks / coverage.
    manifest = list(
        render_benchmark_result_analysis_pack(
            family,
            target_dir / 'generic',
            spec=spec,
            expected_datasets=expected_datasets,
        )
    )

    for table_name, table in (
            ('family_averaged', family),
            ('family_vs_best_single', family_vs_best),
            ('top3_family_by_dataset', top3),
            ('feature_combination_sensitivity', sensitivity),
            ('minirocket_comparison', minirocket),
            ('bottleneck_variants_comparison', bottleneck),
    ):
        manifest.extend(_write_table(table, tables_dir / table_name))

    # Convenience delta using existing helper for documentation parity.
    if not family.empty:
        best_fusion_names = (
            family[family['model_name'].map(classify_model_role) == 'fusion']['model_name'].unique().tolist()
        )
        if best_fusion_names:
            delta = build_dataset_delta_frame(
                family,
                target_model=str(best_fusion_names[0]),
                metric_direction=direction,
            )
            manifest.extend(_write_table(delta, tables_dir / 'example_dataset_delta'))

    summary_path = target_dir / 'fusion_summary.md'
    summary_path.write_text(
        '\n\n'.join(
            [
                '# Fusion-over-raw analysis',
                f'- Primary metric: `{metric_name}`',
                '## Family vs best single',
                dataframe_to_markdown(family_vs_best.head(20)),
                '## MiniRocket comparison',
                dataframe_to_markdown(minirocket.head(20)),
                '## Feature-combination sensitivity',
                dataframe_to_markdown(sensitivity.head(20)),
            ]
        ),
        encoding='utf-8',
    )
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))
    return tuple(manifest)


def _write_table(frame: pd.DataFrame, stem: Path) -> tuple[ArtifactRecord, ...]:
    csv_path = stem.with_suffix('.csv')
    md_path = stem.with_suffix('.md')
    frame.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(frame), encoding='utf-8')
    return (
        ArtifactRecord(kind='table', path=str(csv_path), format='csv'),
        ArtifactRecord(kind='table', path=str(md_path), format='md'),
    )

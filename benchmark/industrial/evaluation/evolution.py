from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from benchmark.industrial.core import ArtifactRecord, ensure_directory
from benchmark.industrial.evaluation.markdown import dataframe_to_markdown


def load_composition_history(
        root: str | Path,
        *,
        dataset_limit: int | None = None,
) -> pd.DataFrame:
    """Load FEDOT/Golem composition history JSON files into a tabular frame."""
    root_path = Path(root)
    if not root_path.exists():
        return _empty_evolution_frame()

    dataset_dirs = [path for path in sorted(root_path.iterdir()) if path.is_dir()]
    if dataset_limit is not None:
        dataset_dirs = dataset_dirs[:dataset_limit]

    rows: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        for json_path in sorted(dataset_dir.rglob('*.json')):
            relative_parts = json_path.relative_to(dataset_dir).parts
            generation = _safe_int(relative_parts[0]) if relative_parts else None
            rows.append(_parse_individual_json(dataset_dir.name, generation, json_path))
    if not rows:
        return _empty_evolution_frame()
    return pd.DataFrame(rows).sort_values(['dataset_name', 'generation', 'fitness_primary'])


def build_evolution_dynamics_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=['dataset_name', 'generation', 'best_fitness', 'mean_fitness', 'pipeline_count'])
    return (
        history.groupby(['dataset_name', 'generation'])
        .agg(
            best_fitness=('fitness_primary', 'min'),
            mean_fitness=('fitness_primary', 'mean'),
            pipeline_count=('pipeline_id', 'nunique'),
            mean_node_count=('node_count', 'mean'),
            mean_depth=('depth', 'mean'),
        )
        .reset_index()
        .sort_values(['dataset_name', 'generation'])
    )


def build_pipeline_complexity_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=['dataset_name', 'pipeline_id', 'node_count', 'depth'])
    return (
        history[
            [
                'dataset_name',
                'generation',
                'pipeline_id',
                'fitness_primary',
                'node_count',
                'depth',
                'operation_chain',
                'computation_time_in_seconds',
            ]
        ]
        .sort_values(['dataset_name', 'fitness_primary', 'node_count'])
        .reset_index(drop=True)
    )


def build_evolution_coverage_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(
            columns=['dataset_count', 'pipeline_count', 'generation_count', 'min_generation', 'max_generation']
        )
    return pd.DataFrame(
        [
            {
                'dataset_count': int(history['dataset_name'].nunique()),
                'pipeline_count': int(history['pipeline_id'].nunique()),
                'generation_count': int(history['generation'].nunique()),
                'min_generation': _safe_int(history['generation'].min()),
                'max_generation': _safe_int(history['generation'].max()),
            }
        ]
    )


def build_operation_frequency_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=['operation_name', 'pipeline_count', 'dataset_count'])
    rows = []
    for _, record in history.iterrows():
        operations = tuple(item.strip() for item in str(record.get('operation_chain', '')).split('->') if item.strip())
        for operation in operations:
            rows.append(
                {
                    'operation_name': operation,
                    'pipeline_id': record.get('pipeline_id'),
                    'dataset_name': record.get('dataset_name'),
                }
            )
    if not rows:
        return pd.DataFrame(columns=['operation_name', 'pipeline_count', 'dataset_count'])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby('operation_name')
        .agg(
            pipeline_count=('pipeline_id', 'nunique'),
            dataset_count=('dataset_name', 'nunique'),
        )
        .reset_index()
        .sort_values(['pipeline_count', 'operation_name'], ascending=[False, True])
        .reset_index(drop=True)
    )


def select_notable_pipelines(
        history: pd.DataFrame,
        *,
        per_dataset: int = 3,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=history.columns)
    return (
        history.sort_values(['dataset_name', 'fitness_primary', 'node_count'], ascending=[True, True, False])
        .groupby('dataset_name')
        .head(per_dataset)
        .reset_index(drop=True)
    )


def render_evolution_analysis_pack(
        root: str | Path,
        output_dir: str | Path,
        *,
        dataset_limit: int | None = None,
        plot_formats: Sequence[str] = ('png',),
) -> tuple[ArtifactRecord, ...]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target_dir = ensure_directory(output_dir)
    tables_dir = ensure_directory(target_dir / 'tables')
    plots_dir = ensure_directory(target_dir / 'plots')
    manifest: list[ArtifactRecord] = []

    history = load_composition_history(root, dataset_limit=dataset_limit)
    coverage = build_evolution_coverage_frame(history)
    dynamics = build_evolution_dynamics_frame(history)
    complexity = build_pipeline_complexity_frame(history)
    operation_frequency = build_operation_frequency_frame(history)
    notable = select_notable_pipelines(history)

    for name, frame in (
            ('coverage', coverage),
            ('composition_history', history),
            ('evolution_dynamics', dynamics),
            ('pipeline_complexity', complexity),
            ('operation_frequency', operation_frequency),
            ('notable_pipelines', notable),
    ):
        manifest.extend(_write_table(frame, tables_dir / name))

    summary_path = target_dir / 'summary.md'
    summary_path.write_text(_summary_markdown(history, dynamics, notable), encoding='utf-8')
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    if not dynamics.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        for dataset_name, group in dynamics.groupby('dataset_name'):
            axis.plot(group['generation'], group['best_fitness'], label=dataset_name, linewidth=1.8)
        axis.set_title('Best fitness by generation')
        axis.set_xlabel('Generation')
        axis.set_ylabel('Best fitness')
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        manifest.extend(_save_figure(figure, plots_dir / 'best_fitness_by_generation', plot_formats))

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.scatter(complexity['node_count'], complexity['fitness_primary'], alpha=0.75)
        axis.set_title('Pipeline complexity vs fitness')
        axis.set_xlabel('Node count')
        axis.set_ylabel('Fitness')
        axis.grid(alpha=0.2)
        manifest.extend(_save_figure(figure, plots_dir / 'complexity_vs_fitness', plot_formats))

    if not operation_frequency.empty:
        figure, axis = plt.subplots(figsize=(10, max(4, 0.28 * min(len(operation_frequency), 25))))
        plot_frame = operation_frequency.head(25).sort_values('pipeline_count', ascending=True)
        axis.barh(plot_frame['operation_name'], plot_frame['pipeline_count'])
        axis.set_title('Most frequent operations in composite pipelines')
        axis.set_xlabel('Pipeline count')
        axis.set_ylabel('Operation')
        axis.grid(alpha=0.2, axis='x')
        manifest.extend(_save_figure(figure, plots_dir / 'operation_frequency', plot_formats))

    return tuple(manifest)


def _parse_individual_json(dataset_name: str, generation: int | None, json_path: Path) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding='utf-8'))
    nodes = _extract_nodes(payload)
    fitness_values = list(_as_iterable(payload.get('fitness', {}).get('_values')))
    metadata = dict(payload.get('metadata') or {})
    node_names = [str(node.get('content', {}).get('name', 'unknown')) for node in nodes]
    return {
        'dataset_name': dataset_name,
        'generation': generation if generation is not None else _safe_int(payload.get('native_generation')),
        'pipeline_id': str(payload.get('uid') or json_path.stem),
        'fitness_primary': float(fitness_values[0]) if fitness_values else float('nan'),
        'fitness_secondary': float(fitness_values[1]) if len(fitness_values) > 1 else None,
        'node_count': len(nodes),
        'depth': _estimate_depth(nodes),
        'operation_chain': ' -> '.join(node_names),
        'computation_time_in_seconds': metadata.get('computation_time_in_seconds'),
        'evaluation_time_iso': metadata.get('evaluation_time_iso'),
        'path': str(json_path),
    }


def _extract_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    graph = dict(payload.get('graph') or {})
    operator = dict(graph.get('operator') or {})
    nodes = operator.get('_nodes') or []
    return [dict(node) for node in nodes if isinstance(node, dict)]


def _estimate_depth(nodes: list[dict[str, Any]]) -> int:
    if not nodes:
        return 0
    parents_by_uid = {
        str(node.get('uid')): tuple(str(parent) for parent in node.get('_nodes_from') or ())
        for node in nodes
    }

    def depth(uid: str, seen: frozenset[str] = frozenset()) -> int:
        if uid in seen:
            return 1
        parents = parents_by_uid.get(uid, ())
        if not parents:
            return 1
        return 1 + max(depth(parent, seen | {uid}) for parent in parents)

    return max(depth(uid) for uid in parents_by_uid)


def _write_table(frame: pd.DataFrame, path_without_suffix: Path) -> tuple[ArtifactRecord, ...]:
    csv_path = path_without_suffix.with_suffix('.csv')
    md_path = path_without_suffix.with_suffix('.md')
    frame.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(frame, index=False) if not frame.empty else 'No rows.', encoding='utf-8')
    return (
        ArtifactRecord(kind='table', path=str(csv_path), format='csv'),
        ArtifactRecord(kind='summary', path=str(md_path), format='md'),
    )


def _summary_markdown(history: pd.DataFrame, dynamics: pd.DataFrame, notable: pd.DataFrame) -> str:
    lines = [
        '# Evolution And Composition Analysis',
        '',
        f'- Pipelines: `{len(history)}`',
        f'- Datasets: `{history["dataset_name"].nunique() if not history.empty else 0}`',
        f'- Generations: `{dynamics["generation"].nunique() if not dynamics.empty else 0}`',
        f'- Dataset limit: `none`' if not history.empty else '- Dataset limit: `not available`',
        '',
        '## Notable Pipelines',
        '',
        dataframe_to_markdown(
            notable[['dataset_name', 'generation', 'pipeline_id', 'fitness_primary', 'node_count', 'depth',
                     'operation_chain']],
            index=False,
        ) if not notable.empty else 'No notable pipelines.',
    ]
    return '\n'.join(lines)


def _save_figure(figure, path_without_suffix: Path, plot_formats: Sequence[str]) -> tuple[ArtifactRecord, ...]:
    import matplotlib.pyplot as plt

    manifest: list[ArtifactRecord] = []
    for extension in plot_formats:
        path = path_without_suffix.with_suffix(f'.{extension}')
        figure.savefig(path, dpi=200, bbox_inches='tight')
        manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
    plt.close(figure)
    return tuple(manifest)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _empty_evolution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            'dataset_name',
            'generation',
            'pipeline_id',
            'fitness_primary',
            'fitness_secondary',
            'node_count',
            'depth',
            'operation_chain',
            'computation_time_in_seconds',
            'evaluation_time_iso',
            'path',
        ]
    )


__all__ = [
    'build_evolution_dynamics_frame',
    'build_evolution_coverage_frame',
    'build_operation_frequency_frame',
    'build_pipeline_complexity_frame',
    'load_composition_history',
    'render_evolution_analysis_pack',
    'select_notable_pipelines',
]

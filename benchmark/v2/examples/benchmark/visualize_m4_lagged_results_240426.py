from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
V2_ROOT = PROJECT_ROOT / 'benchmark' / 'v2'
for candidate in (str(PROJECT_ROOT), str(V2_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from benchmark.v2.result_visualization import visualize_forecasting_progress_items
except Exception:
    from result_visualization import visualize_forecasting_progress_items


DEFAULT_ITEMS_DIR = (
    Path(__file__).resolve().parent
    / 'results'
    / 'v2_demo'
    / 'm4_regime_suite_220426'
    / 'm4_regime_suite_6cb78aebac'
    / 'progress'
    / 'items'
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_ITEMS_DIR.parent.parent
    / 'visualizations'
    / 'lagged_forecaster_m4_daily'
)


def _parse_max_series_plots(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {'none', 'all', '*'}:
        return None
    return int(normalized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Визуализация и аналитика результатов lagged_forecaster на M4 Daily benchmark.'
    )
    parser.add_argument('--items-dir', type=Path, default=DEFAULT_ITEMS_DIR, help='Папка progress/items с checkpoint JSON.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Папка для сохранения графиков и сводных таблиц.',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='lagged_forecaster',
        help='Имя модели для фильтрации item-файлов.',
    )
    parser.add_argument(
        '--series-ids',
        nargs='*',
        default=(),
        help='Опциональный список series_id для детальных per-series графиков.',
    )
    parser.add_argument(
        '--max-series-plots',
        type=_parse_max_series_plots,
        default=None,
        help='Если series_ids не заданы, сколько рядов визуализировать детально.',
    )
    parser.add_argument(
        '--plot-formats',
        nargs='*',
        default=('png', 'svg'),
        help='Форматы сохранения графиков.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = visualize_forecasting_progress_items(
        args.items_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        series_ids=tuple(args.series_ids),
        max_series_plots=args.max_series_plots,
        plot_formats=tuple(args.plot_formats),
    )

    print(f'Items analysed: {len(result.items_frame)}')
    print(f'Relative gain rows: {len(result.relative_gain_frame)}')
    print(f'Artifacts saved: {len(result.artifact_manifest)}')
    print(f'Output dir: {args.output_dir}')


if __name__ == '__main__':
    main()

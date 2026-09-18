from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.industrial.evaluation import render_okhs_smoothing_acceptance_pack
from benchmark.industrial.visualization import visualize_forecasting_progress_items


def render_okhs_forecasting_progress(
    items_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    series_ids: tuple[str, ...] = (),
    max_series_plots: int | None = 3,
) -> Any:
    return visualize_forecasting_progress_items(
        items_dir=items_dir,
        output_dir=output_dir,
        model_name=model_name,
        series_ids=series_ids,
        max_series_plots=max_series_plots,
    )


def render_okhs_acceptance_pack(result: Any, output_dir: str | Path) -> tuple[Any, ...]:
    return tuple(render_okhs_smoothing_acceptance_pack(result, Path(output_dir)))


__all__ = ["render_okhs_acceptance_pack", "render_okhs_forecasting_progress"]

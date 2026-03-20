from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fedot_ind.core.models.automl.okhs_benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkModelSpec,
    BenchmarkReport,
    QPolicy,
    QSelectionSpec,
    RollingForecastConfig,
    RollingForecastReport,
    UncertaintyConfig,
    UncertaintyReport,
    benchmark_report_to_dict,
    build_naive_drift_spec,
    build_naive_last_value_spec,
    build_naive_mean_spec,
    build_okhs_forecaster_spec,
    build_okhs_q_orchestrated_spec,
    build_uncertainty_report,
    render_benchmark_markdown,
    render_rolling_forecast_markdown,
    render_uncertainty_markdown,
    rolling_report_to_dict,
    run_benchmark,
    run_rolling_forecast,
    uncertainty_report_to_dict,
)


@dataclass(frozen=True)
class OKHSProductConfig:
    dataset_name: str
    series: np.ndarray
    forecast_horizon: int
    benchmark_metrics: tuple[str, ...] = ('mae', 'rmse')
    benchmark_validation_fraction: float = 0.2
    benchmark_test_fraction: float = 0.2
    rolling_config: RollingForecastConfig | None = None
    uncertainty_config: UncertaintyConfig | None = None
    q_selection: QSelectionSpec | None = None
    okhs_window_size: int = 20
    okhs_n_modes: int = 5
    include_baselines: tuple[str, ...] = ('naive_last_value', 'naive_mean', 'naive_drift')


@dataclass(frozen=True)
class OKHSProductResult:
    benchmark_report: BenchmarkReport
    rolling_report: RollingForecastReport | None = None
    uncertainty_report: UncertaintyReport | None = None


def build_baseline_specs(names: tuple[str, ...]) -> tuple[BenchmarkModelSpec, ...]:
    builders = {
        'naive_last_value': build_naive_last_value_spec,
        'naive_mean': build_naive_mean_spec,
        'naive_drift': build_naive_drift_spec,
    }
    specs: list[BenchmarkModelSpec] = []
    for name in names:
        if name not in builders:
            raise ValueError(f'Unsupported baseline model: {name}')
        specs.append(builders[name]())
    return tuple(specs)


def build_okhs_product_model_specs(
        forecast_horizon: int,
        q_selection: QSelectionSpec | None = None,
        window_size: int = 20,
        n_modes: int = 5,
        include_baselines: tuple[str, ...] = ('naive_last_value', 'naive_mean', 'naive_drift'),
) -> tuple[BenchmarkModelSpec, ...]:
    if q_selection is None:
        okhs_spec = build_okhs_forecaster_spec(
            name='okhs_dmd',
            q=0.7,
            forecast_horizon=forecast_horizon,
            n_modes=n_modes,
            q_policy=QPolicy.FIXED,
            window_size=window_size,
        )
    else:
        okhs_spec = build_okhs_q_orchestrated_spec(
            name='okhs_dmd_q_orchestrated',
            q_selection=q_selection,
            forecast_horizon=forecast_horizon,
            n_modes=n_modes,
            window_size=window_size,
        )

    return (okhs_spec,) + build_baseline_specs(include_baselines)


def run_okhs_product_suite(config: OKHSProductConfig) -> OKHSProductResult:
    normalized_series = np.asarray(config.series, dtype=float).reshape(-1)
    model_specs = build_okhs_product_model_specs(
        forecast_horizon=config.forecast_horizon,
        q_selection=config.q_selection,
        window_size=config.okhs_window_size,
        n_modes=config.okhs_n_modes,
        include_baselines=config.include_baselines,
    )
    benchmark_report = run_benchmark(
        BenchmarkConfig(
            datasets=(
                BenchmarkDataset(
                    name=config.dataset_name,
                    series=normalized_series,
                    forecast_horizon=config.forecast_horizon,
                    tags=('okhs_product',),
                ),
            ),
            model_specs=model_specs,
            metrics=config.benchmark_metrics,
            validation_fraction=config.benchmark_validation_fraction,
            test_fraction=config.benchmark_test_fraction,
        )
    )

    rolling_report = None
    uncertainty_report = None
    if config.rolling_config is not None:
        okhs_model_spec = model_specs[0]
        rolling_report = run_rolling_forecast(
            series=normalized_series,
            model_spec=okhs_model_spec,
            config=config.rolling_config,
        )
        if config.uncertainty_config is not None:
            uncertainty_report = build_uncertainty_report(
                rolling_report=rolling_report,
                config=config.uncertainty_config,
            )

    return OKHSProductResult(
        benchmark_report=benchmark_report,
        rolling_report=rolling_report,
        uncertainty_report=uncertainty_report,
    )


def okhs_product_result_to_dict(result: OKHSProductResult) -> dict[str, Any]:
    return {
        'benchmark_report': benchmark_report_to_dict(result.benchmark_report),
        'rolling_report': (
            rolling_report_to_dict(result.rolling_report)
            if result.rolling_report is not None
            else None
        ),
        'uncertainty_report': (
            uncertainty_report_to_dict(result.uncertainty_report)
            if result.uncertainty_report is not None
            else None
        ),
    }


def render_okhs_product_markdown(result: OKHSProductResult) -> str:
    sections = [
        '# OKHS Product Report',
        '',
        '## Benchmark',
        render_benchmark_markdown(result.benchmark_report),
    ]

    if result.rolling_report is not None:
        sections.extend(['', '## Rolling Forecast', render_rolling_forecast_markdown(result.rolling_report)])

    if result.uncertainty_report is not None:
        sections.extend(['', '## Uncertainty', render_uncertainty_markdown(result.uncertainty_report)])

    return '\n'.join(sections)

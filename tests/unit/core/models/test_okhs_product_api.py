import numpy as np
import pytest

from fedot_ind.core.models.automl import okhs_product as okhs_product_module
from fedot_ind.core.models.automl.okhs_benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    RollingForecastConfig,
    UncertaintyConfig,
    run_benchmark,
)
from fedot_ind.core.models.automl.okhs_product import (
    OKHSProductConfig,
    build_baseline_specs,
    build_okhs_product_model_specs,
    okhs_product_result_to_dict,
    render_okhs_product_markdown,
    run_okhs_product_suite,
)


def test_build_baseline_specs_supports_mean_and_drift():
    specs = build_baseline_specs(('naive_last_value', 'naive_mean', 'naive_drift'))

    assert [spec.name for spec in specs] == ['naive_last_value', 'naive_mean', 'naive_drift']
    assert all('baseline' in spec.tags for spec in specs)


def test_baseline_specs_participate_in_benchmark():
    report = run_benchmark(
        BenchmarkConfig(
            datasets=(
                BenchmarkDataset(
                    name='trend_demo',
                    series=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float),
                    forecast_horizon=1,
                ),
            ),
            model_specs=build_baseline_specs(('naive_last_value', 'naive_mean', 'naive_drift')),
            metrics=('mae',),
            validation_fraction=0.17,
            test_fraction=0.17,
        )
    )

    assert len(report.runs) == 3
    assert report.leaderboard[0]['model_name'] == 'naive_drift'


def test_run_okhs_product_suite_returns_all_reports(monkeypatch):
    class FakeProductSpec:
        def __init__(self, name):
            self.name = name

    def fake_build_specs(**kwargs):
        return (
            FakeProductSpec('okhs_dmd_q_orchestrated'),
            FakeProductSpec('naive_last_value'),
        )

    def fake_run_benchmark(config):
        assert config.datasets[0].name == 'demo_series'
        return {'kind': 'benchmark', 'model_names': [spec.name for spec in config.model_specs]}

    def fake_run_rolling_forecast(series, model_spec, config):
        assert model_spec.name == 'okhs_dmd_q_orchestrated'
        return {'kind': 'rolling', 'series_length': len(series), 'horizon': config.forecast_horizon}

    def fake_build_uncertainty_report(rolling_report, config):
        assert rolling_report['kind'] == 'rolling'
        return {'kind': 'uncertainty', 'confidence_level': config.confidence_level}

    monkeypatch.setattr(okhs_product_module, 'build_okhs_product_model_specs', fake_build_specs)
    monkeypatch.setattr(okhs_product_module, 'run_benchmark', fake_run_benchmark)
    monkeypatch.setattr(okhs_product_module, 'run_rolling_forecast', fake_run_rolling_forecast)
    monkeypatch.setattr(okhs_product_module, 'build_uncertainty_report', fake_build_uncertainty_report)
    monkeypatch.setattr(okhs_product_module, 'benchmark_report_to_dict', lambda report: report)
    monkeypatch.setattr(okhs_product_module, 'rolling_report_to_dict', lambda report: report)
    monkeypatch.setattr(okhs_product_module, 'uncertainty_report_to_dict', lambda report: report)
    monkeypatch.setattr(okhs_product_module, 'render_benchmark_markdown', lambda report: 'benchmark-markdown')
    monkeypatch.setattr(okhs_product_module, 'render_rolling_forecast_markdown', lambda report: 'rolling-markdown')
    monkeypatch.setattr(okhs_product_module, 'render_uncertainty_markdown', lambda report: 'uncertainty-markdown')

    result = run_okhs_product_suite(
        OKHSProductConfig(
            dataset_name='demo_series',
            series=np.arange(12, dtype=float),
            forecast_horizon=2,
            rolling_config=RollingForecastConfig(forecast_horizon=2),
            uncertainty_config=UncertaintyConfig(confidence_level=0.9),
        )
    )

    assert result.benchmark_report['kind'] == 'benchmark'
    assert result.rolling_report['kind'] == 'rolling'
    assert result.uncertainty_report['kind'] == 'uncertainty'

    payload = okhs_product_result_to_dict(result)
    assert payload['benchmark_report']['kind'] == 'benchmark'
    assert payload['uncertainty_report']['confidence_level'] == pytest.approx(0.9)

    markdown = render_okhs_product_markdown(result)
    assert '# OKHS Product Report' in markdown
    assert 'benchmark-markdown' in markdown
    assert 'uncertainty-markdown' in markdown


def test_build_okhs_product_model_specs_includes_okhs_and_requested_baselines(monkeypatch):
    monkeypatch.setattr(
        okhs_product_module,
        'build_okhs_forecaster_spec',
        lambda **kwargs: type('Spec', (), {'name': 'okhs_dmd', 'tags': ('okhs',)})(),
    )

    specs = build_okhs_product_model_specs(
        forecast_horizon=3,
        include_baselines=('naive_mean', 'naive_drift'),
    )

    assert [spec.name for spec in specs] == ['okhs_dmd', 'naive_mean', 'naive_drift']

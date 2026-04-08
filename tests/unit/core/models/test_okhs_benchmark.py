import numpy as np
import pytest

from fedot_ind.core.models.automl import okhs_benchmark as okhs_benchmark_module
from fedot_ind.core.models.automl.okhs_benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkModelSpec,
    ContextWindowPolicy,
    build_uncertainty_report,
    QPolicy,
    QSelectionSpec,
    RefitPolicy,
    benchmark_report_to_dict,
    build_holdout_split,
    build_naive_last_value_spec,
    build_okhs_forecaster_spec,
    build_okhs_q_orchestrated_spec,
    rolling_report_to_dict,
    render_benchmark_markdown,
    render_rolling_forecast_markdown,
    render_uncertainty_markdown,
    RollingForecastConfig,
    UncertaintyConfig,
    uncertainty_report_to_dict,
    run_benchmark,
    run_rolling_forecast,
    validate_benchmark_config,
)


def test_build_holdout_split_creates_expected_segments():
    split = build_holdout_split(np.arange(10, dtype=float), validation_fraction=0.2, test_fraction=0.2)

    assert split.train.tolist() == [0, 1, 2, 3, 4, 5]
    assert split.validation.tolist() == [6, 7]
    assert split.test.tolist() == [8, 9]


def test_validate_benchmark_config_rejects_invalid_fractions():
    config = BenchmarkConfig(
        datasets=(BenchmarkDataset(name='demo', series=np.arange(20, dtype=float), forecast_horizon=2),),
        model_specs=(build_naive_last_value_spec(),),
        validation_fraction=0.6,
        test_fraction=0.5,
    )

    with pytest.raises(ValueError, match='less than 1'):
        validate_benchmark_config(config)


def test_run_benchmark_builds_report_and_leaderboard():
    class PerfectModel:
        def fit(self, series):
            self.last_seen_ = np.asarray(series)

        def predict(self, horizon):
            return np.array([8.0] * horizon)

    class OffsetModel:
        def fit(self, series):
            self.last_seen_ = np.asarray(series)

        def predict(self, horizon):
            return np.array([9.0] * horizon)

    perfect_spec = BenchmarkModelSpec(
        name='perfect',
        factory=PerfectModel,
        fit_fn=lambda model, series: model.fit(series),
        predict_fn=lambda model, context, horizon: model.predict(horizon),
        tags=('test',),
    )
    offset_spec = BenchmarkModelSpec(
        name='offset',
        factory=OffsetModel,
        fit_fn=lambda model, series: model.fit(series),
        predict_fn=lambda model, context, horizon: model.predict(horizon),
    )

    dataset = BenchmarkDataset(
        name='constant_tail',
        series=np.array([1, 2, 3, 4, 5, 6, 8, 8, 8, 8], dtype=float),
        forecast_horizon=2,
    )
    report = run_benchmark(
        BenchmarkConfig(
            datasets=(dataset,),
            model_specs=(perfect_spec, offset_spec),
            metrics=('mae', 'rmse'),
            validation_fraction=0.2,
            test_fraction=0.2,
        )
    )

    assert len(report.runs) == 2
    assert report.leaderboard[0]['model_name'] == 'perfect'
    assert report.runs[0].test_metrics['mae'] == pytest.approx(0.0)
    assert report.runs[1].test_metrics['mae'] == pytest.approx(1.0)

    report_dict = benchmark_report_to_dict(report)
    assert report_dict['leaderboard'][0]['model_name'] == 'perfect'

    markdown = render_benchmark_markdown(report)
    assert '# OKHS Benchmark Report' in markdown
    assert 'constant_tail / perfect' in markdown


def test_build_okhs_forecaster_spec_adapts_public_forecaster_api(monkeypatch):
    observed = {}

    class FakeForecaster:
        def __init__(self, q, forecast_horizon, n_modes, method, q_policy):
            observed['init'] = {
                'q': q,
                'forecast_horizon': forecast_horizon,
                'n_modes': n_modes,
                'method': method,
                'q_policy': q_policy,
            }
            self.forecast_horizon = forecast_horizon

        def fit(self, series, window_size=20):
            observed['fit'] = {
                'series': np.asarray(series).tolist(),
                'window_size': window_size,
            }

        def predict(self, context_series):
            observed['predict'] = {
                'context_series': np.asarray(context_series).tolist(),
                'forecast_horizon': self.forecast_horizon,
            }
            return np.full(self.forecast_horizon, 3.5, dtype=float)

    monkeypatch.setattr(okhs_benchmark_module, 'OKHSForecaster', FakeForecaster)

    spec = build_okhs_forecaster_spec(
        name='okhs_dmd',
        q=0.65,
        forecast_horizon=4,
        n_modes=3,
        window_size=7,
    )
    model = spec.factory()
    spec.fit_fn(model, np.arange(12, dtype=float))
    forecast = spec.predict_fn(model, np.arange(12, dtype=float), 2)

    assert observed['init']['q'] == pytest.approx(0.65)
    assert observed['fit']['window_size'] == 7
    assert observed['predict']['forecast_horizon'] == 2
    assert forecast.tolist() == [3.5, 3.5]


def test_fixed_q_orchestrated_spec_records_decision_trace(monkeypatch):
    observed = []

    class FakeForecaster:
        def __init__(self, q, forecast_horizon, n_modes, method, q_policy):
            self.q = float(q)
            self.forecast_horizon = forecast_horizon

        def fit(self, series, window_size=20):
            observed.append(('fit', self.q, window_size, len(series)))

        def predict(self, context_series):
            observed.append(('predict', self.q, len(context_series), self.forecast_horizon))
            return np.full(self.forecast_horizon, self.q, dtype=float)

    monkeypatch.setattr(okhs_benchmark_module, 'OKHSForecaster', FakeForecaster)

    spec = build_okhs_q_orchestrated_spec(
        name='okhs_fixed_q',
        q_selection=QSelectionSpec(policy=QPolicy.FIXED, fixed_q=0.55),
        forecast_horizon=3,
        window_size=5,
    )
    model = spec.factory()
    spec.fit_fn(model, np.arange(14, dtype=float))
    forecast = spec.predict_fn(model, np.arange(14, dtype=float), 2)
    metadata = spec.metadata_fn(model)

    assert forecast.tolist() == [0.55, 0.55]
    assert metadata['policy'] == 'fixed'
    assert metadata['selected_q'] == pytest.approx(0.55)
    assert metadata['candidate_scores'] == []
    assert observed[0][1] == pytest.approx(0.55)


def test_data_driven_q_orchestrated_spec_records_selector_diagnostics(monkeypatch):
    class FakeSelector:
        def suggest_q_based_on_autocorrelation(self, series):
            return 0.8

        def suggest_q_based_on_frequency(self, trajectories):
            return 0.4

    class FakeForecaster:
        def __init__(self, q, forecast_horizon, n_modes, method, q_policy):
            self.q = float(q)
            self.forecast_horizon = forecast_horizon

        def fit(self, series, window_size=20):
            self.series_ = np.asarray(series)

        def predict(self, context_series):
            return np.full(self.forecast_horizon, self.q, dtype=float)

    monkeypatch.setattr(okhs_benchmark_module, 'OKHSForecaster', FakeForecaster)

    spec = build_okhs_q_orchestrated_spec(
        name='okhs_data_driven_q',
        q_selection=QSelectionSpec(
            policy=QPolicy.DATA_DRIVEN,
            selector=FakeSelector(),
        ),
        forecast_horizon=2,
        window_size=4,
    )
    model = spec.factory()
    spec.fit_fn(model, np.arange(12, dtype=float))
    metadata = spec.metadata_fn(model)

    assert metadata['policy'] == 'data_driven'
    assert metadata['selected_q'] == pytest.approx(0.6)
    assert metadata['diagnostics']['autocorrelation_q'] == pytest.approx(0.8)
    assert metadata['diagnostics']['frequency_q'] == pytest.approx(0.4)


def test_search_q_orchestrated_spec_selects_best_q_in_benchmark(monkeypatch):
    class FakeForecaster:
        def __init__(self, q, forecast_horizon, n_modes, method, q_policy):
            self.q = float(q)
            self.forecast_horizon = forecast_horizon

        def fit(self, series, window_size=20):
            self.series_ = np.asarray(series)

        def predict(self, context_series):
            del context_series
            return np.full(self.forecast_horizon, self.q, dtype=float)

    monkeypatch.setattr(okhs_benchmark_module, 'OKHSForecaster', FakeForecaster)

    report = run_benchmark(
        BenchmarkConfig(
            datasets=(
                BenchmarkDataset(
                    name='search_demo',
                    series=np.array([0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=float),
                    forecast_horizon=2,
                ),
            ),
            model_specs=(
                build_okhs_q_orchestrated_spec(
                    name='okhs_search_q',
                    q_selection=QSelectionSpec(
                        policy='search',
                        q_grid=(0.3, 0.6, 0.9),
                        selection_metric='mae',
                        validation_fraction=0.25,
                    ),
                    forecast_horizon=2,
                    window_size=2,
                ),
            ),
            metrics=('mae',),
            validation_fraction=0.25,
            test_fraction=0.25,
        )
    )

    run = report.runs[0]
    assert run.validation_metadata['policy'] == 'search'
    assert run.validation_metadata['selected_q'] == pytest.approx(0.6)
    assert len(run.validation_metadata['candidate_scores']) == 3
    assert run.test_metadata['selected_q'] == pytest.approx(0.6)

    markdown = render_benchmark_markdown(report)
    assert 'policy=search' in markdown


def test_run_rolling_forecast_respects_periodic_refit_policy():
    fit_calls = []
    predict_calls = []

    class FakeRollingModel:
        def fit(self, series):
            fit_calls.append(np.asarray(series).tolist())

        def predict(self, horizon):
            predict_calls.append(horizon)
            return np.full(horizon, 1.0, dtype=float)

    spec = BenchmarkModelSpec(
        name='rolling_stub',
        factory=FakeRollingModel,
        fit_fn=lambda model, series: model.fit(series),
        predict_fn=lambda model, context, horizon: model.predict(horizon),
    )
    report = run_rolling_forecast(
        series=np.arange(10, dtype=float),
        model_spec=spec,
        config=RollingForecastConfig(
            forecast_horizon=2,
            step_size=1,
            refit_policy=RefitPolicy.PERIODIC,
            update_frequency=2,
            metrics=('mae',),
        ),
    )

    assert [step.refit_performed for step in report.steps[:4]] == [True, False, True, False]
    assert len(fit_calls) == 4
    assert len(predict_calls) == len(report.steps)
    assert report.aggregate_metrics['mae'] >= 0


def test_run_rolling_forecast_uses_fixed_rolling_window():
    fit_calls = []

    class FakeRollingModel:
        def fit(self, series):
            fit_calls.append(np.asarray(series).tolist())

        def predict(self, horizon):
            return np.zeros(horizon, dtype=float)

    spec = BenchmarkModelSpec(
        name='rolling_window_stub',
        factory=FakeRollingModel,
        fit_fn=lambda model, series: model.fit(series),
        predict_fn=lambda model, context, horizon: model.predict(horizon),
    )
    report = run_rolling_forecast(
        series=np.arange(9, dtype=float),
        model_spec=spec,
        config=RollingForecastConfig(
            forecast_horizon=2,
            step_size=2,
            context_policy=ContextWindowPolicy.ROLLING,
            rolling_window_size=4,
            refit_policy=RefitPolicy.ALWAYS,
            metrics=('mae',),
        ),
    )

    assert fit_calls[0] == [0.0, 1.0, 2.0, 3.0]
    assert fit_calls[1] == [2.0, 3.0, 4.0, 5.0]
    assert all(step.metadata['context_policy'] == 'rolling' for step in report.steps)


def test_run_rolling_forecast_supports_drift_refit_and_report_rendering():
    fit_calls = []

    class DriftAwareModel:
        def fit(self, series):
            fit_calls.append(np.asarray(series).tolist())

        def predict(self, horizon):
            return np.zeros(horizon, dtype=float)

    spec = BenchmarkModelSpec(
        name='drift_stub',
        factory=DriftAwareModel,
        fit_fn=lambda model, series: model.fit(series),
        predict_fn=lambda model, context, horizon: model.predict(horizon),
        metadata_fn=lambda model: {'stub': True},
    )
    report = run_rolling_forecast(
        series=np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], dtype=float),
        model_spec=spec,
        config=RollingForecastConfig(
            forecast_horizon=1,
            step_size=1,
            refit_policy=RefitPolicy.DRIFT,
            drift_threshold=0.5,
            metrics=('mae',),
        ),
    )

    assert [step.refit_performed for step in report.steps[:3]] == [True, False, True]
    assert report.steps[0].metadata['stub'] is True

    payload = rolling_report_to_dict(report)
    assert payload['model_name'] == 'drift_stub'

    markdown = render_rolling_forecast_markdown(report)
    assert '# OKHS Rolling Forecast Report' in markdown
    assert 'refit_policy: drift' in markdown


def test_build_uncertainty_report_uses_residual_history_and_marks_early_steps():
    class OffsetModel:
        def fit(self, series):
            self.last_ = float(np.asarray(series)[-1])

        def predict(self, horizon):
            return np.full(horizon, self.last_, dtype=float)

    spec = BenchmarkModelSpec(
        name='offset_stub',
        factory=OffsetModel,
        fit_fn=lambda model, series: model.fit(series),
        predict_fn=lambda model, context, horizon: model.predict(horizon),
    )
    rolling_report = run_rolling_forecast(
        series=np.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0], dtype=float),
        model_spec=spec,
        config=RollingForecastConfig(
            forecast_horizon=1,
            step_size=1,
            refit_policy=RefitPolicy.ALWAYS,
            metrics=('mae',),
        ),
    )

    uncertainty_report = build_uncertainty_report(
        rolling_report,
        UncertaintyConfig(confidence_level=0.9, min_history=2, error_floor=0.1),
    )

    assert uncertainty_report.steps[0].quality_flags == ('insufficient_history',)
    assert uncertainty_report.steps[1].interval.center == rolling_report.steps[1].forecast_values
    assert uncertainty_report.steps[1].residual_scale >= 0.1
    assert uncertainty_report.diagnostics['method'] == 'residual_std'


def test_uncertainty_report_serialization_and_markdown_rendering():
    class ZeroModel:
        def fit(self, series):
            self.series_ = np.asarray(series)

        def predict(self, horizon):
            return np.zeros(horizon, dtype=float)

    spec = BenchmarkModelSpec(
        name='zero_stub',
        factory=ZeroModel,
        fit_fn=lambda model, series: model.fit(series),
        predict_fn=lambda model, context, horizon: model.predict(horizon),
    )
    rolling_report = run_rolling_forecast(
        series=np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
        model_spec=spec,
        config=RollingForecastConfig(
            forecast_horizon=1,
            step_size=1,
            refit_policy=RefitPolicy.NEVER,
            metrics=('mae',),
        ),
    )
    uncertainty_report = build_uncertainty_report(
        rolling_report,
        UncertaintyConfig(confidence_level=0.95, min_history=1, width_warning_threshold=0.2),
    )

    payload = uncertainty_report_to_dict(uncertainty_report)
    assert payload['model_name'] == 'zero_stub'
    assert payload['steps'][0]['interval']['center'] == [0.0]

    markdown = render_uncertainty_markdown(uncertainty_report)
    assert '# OKHS Uncertainty Report' in markdown
    assert 'confidence_level: 0.95' in markdown

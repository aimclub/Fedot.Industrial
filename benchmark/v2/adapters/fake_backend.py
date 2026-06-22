from __future__ import annotations

import numpy as np

from .automl_common import (
    CommonAutoMLAdapter,
    AutoMLBudget,
    AutoMLResourceReport,
)

from benchmark.v2.core import ForecastingDatasetRecord


def make_record():
    return ForecastingDatasetRecord(
        benchmark="test",
        dataset_name="test",
        subset="default",
        series_id="s1",
        frequency="D",
        forecast_horizon=3,
        seasonal_period=1,
        train_values=(1, 2, 3, 4, 5),
        test_values=(6, 7, 8),
        known_future_covariates={},
        observed_past_covariates={},
        static_covariates={},
        panel_ids={},
        hierarchy={},
        metadata={}
    )


record = make_record()


class FakeAutoMLBackend(CommonAutoMLAdapter):
    backend_name = "fake_automl"

    _quantile_supported = True

    def _import_backend(self):
        return object()

    def _prepare_data(self, dataset_record):
        return {
            "train": np.asarray(dataset_record.train_values, dtype=float),
            "dataset_record": dataset_record,
        }

    def _fit(
        self,
        data,
        budget: AutoMLBudget,
        kwargs,
    ):
        return {
            "last_value": float(data["train"][-1]),
            "budget": budget,
        }

    def _predict(
        self,
        model,
        data,
        forecast_horizon: int,
    ) -> np.ndarray:
        return np.full(
            forecast_horizon,
            model["last_value"],
            dtype=float,
        )

    def _predict_quantiles(self, model, data, forecast_horizon: int, quantiles) -> np.ndarray:
        base = model["last_value"]
        result = np.zeros((forecast_horizon, len(quantiles)), dtype=float)
        for j, q in enumerate(quantiles):
            result[:, j] = base + (q - 0.5)
        return result

    def supports_quantiles(self) -> bool:
        return True

    def _extract_report(
        self,
        model,
        start_time,
        end_time,
        kwargs,
    ) -> AutoMLResourceReport:

        return AutoMLResourceReport(
            wall_clock_sec=end_time - start_time,
            models_fitted=3,
            failed_trials=1,
            total_trials=4,
            quantile_support=True,
            memory_usage_mb=10.0,
        )


def test_availability():
    adapter = FakeAutoMLBackend()

    status, _ = adapter.availability()

    assert status.name == "SUCCESS"


def test_budget_propagation():
    budget = AutoMLBudget(
        time_limit_sec=60,
        trial_limit=10,
        random_seed=123,
    )

    adapter = FakeAutoMLBackend(
        budget=budget,
    )

    assert adapter.budget.time_limit_sec == 60
    assert adapter.budget.trial_limit == 10
    assert adapter.budget.random_seed == 123


def test_forecast():
    adapter = FakeAutoMLBackend()

    forecast, metadata = adapter.forecast(record)

    assert len(forecast) == record.forecast_horizon

    assert metadata["models_fitted"] == 3
    assert metadata["failed_trials"] == 1
    assert metadata["quantile_support"] is True


def test_quantiles():
    adapter = FakeAutoMLBackend()

    _, metadata = adapter.forecast(record)

    assert "quantiles" in metadata

    assert "q0.10" in metadata["quantiles"]
    assert "q0.50" in metadata["quantiles"]
    assert "q0.90" in metadata["quantiles"]


def test_resource_report():
    adapter = FakeAutoMLBackend()

    _, metadata = adapter.forecast(record)

    report = metadata["resource_report"]

    assert "wall_clock_sec" in report
    assert "models_fitted" in report
    assert "failed_trials" in report
    assert "quantile_support" in report

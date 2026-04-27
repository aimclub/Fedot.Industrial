from __future__ import annotations

import numpy as np

from benchmark.v2 import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, TaskType
from benchmark.v2 import progress as progress_module
from benchmark.v2.api import run_forecasting_benchmark_suite


class FakeTqdmBar:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get('total', 0)
        self.updated = 0
        self.postfix_calls = []
        self.closed = False

    def refresh(self):
        return None

    def set_postfix(self, payload, refresh=False):
        self.postfix_calls.append((payload, refresh))

    def update(self, amount):
        self.updated += amount

    def close(self):
        self.closed = True


def _toy_records() -> list[dict]:
    base = np.linspace(1.0, 12.0, num=12)
    return [
        {
            'series_id': 'toy_1',
            'values': (base + 0.5 * np.sin(np.arange(12))).tolist(),
            'horizon': 3,
            'frequency': 'monthly',
            'seasonal_period': 3,
            'dataset_name': 'toy_dataset',
        }
    ]


def test_progress_monitor_is_used_by_forecasting_suite(monkeypatch, tmp_path) -> None:
    fake_bar = FakeTqdmBar()
    writes: list[str] = []

    monkeypatch.setattr(progress_module, 'TQDM_FACTORY', lambda *args, **kwargs: fake_bar)
    monkeypatch.setattr(progress_module, 'TQDM_WRITE', lambda message: writes.append(message))

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_dataset',
                subset='monthly',
                adapter_options={
                    'records': _toy_records(),
                    'forecast_horizon': 3,
                    'seasonal_period': 3,
                },
            ),
        ),
        models=(
            ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),
            ModelSpec(adapter_name='autogluon', display_name='AutoGluon', optional=True),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='progress_demo', primary_metric='mae', show_progress=True),
    )

    result = run_forecasting_benchmark_suite(config)

    assert any(record.model_name == 'NaiveLastValue' for record in result.run_records)
    assert fake_bar.total == 2
    assert fake_bar.updated == 2
    assert fake_bar.closed is True
    assert writes
    assert any('dataset=toy_dataset' in message for message in writes)
    assert any('model_summary dataset=toy_dataset model=NaiveLastValue' in message for message in writes)
    assert any('dataset_summary dataset=toy_dataset' in message for message in writes)

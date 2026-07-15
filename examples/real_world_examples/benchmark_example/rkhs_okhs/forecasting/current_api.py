from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.industrial import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, TaskType

FORECASTING_DIR = Path(__file__).resolve().parent
MONASH_BITCOIN_SAMPLE = FORECASTING_DIR / "MonashBitcoin_30.csv"


def build_okhs_forecasting_suite_config(
    output_dir: str | Path | None = None,
    *,
    persist_on_run: bool = False,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark="in_memory",
                dataset_name="monash_bitcoin_okhs_sample",
                subset="daily",
                adapter_options={
                    "records": (_monash_bitcoin_record(),),
                    "forecast_horizon": 5,
                    "seasonal_period": 1,
                },
            ),
        ),
        models=(
            ModelSpec(adapter_name="naive_last_value", display_name="NaiveLastValue"),
            ModelSpec(
                adapter_name="okhs",
                display_name="OKHSForecaster",
                optional=True,
                params={"method": "dmd", "kernel": "periodic"},
            ),
        ),
        metrics=("rmse", "mae", "smape"),
        artifact_spec=ArtifactSpec(
            output_dir=str(output_dir or FORECASTING_DIR / "results"),
            persist_on_run=persist_on_run,
            render_publication_pack=True,
        ),
        run_spec=RunSpec(run_name="rkhs_okhs_forecasting", primary_metric="rmse", show_progress=False),
    )


def config_summary(config: BenchmarkSuiteConfig) -> dict[str, object]:
    return {
        "task_type": config.task_type.value,
        "datasets": [dataset.dataset_name for dataset in config.datasets],
        "models": [model.display_name for model in config.models],
        "persist_on_run": config.artifact_spec.persist_on_run,
    }


def _monash_bitcoin_record() -> dict[str, object]:
    if MONASH_BITCOIN_SAMPLE.exists():
        frame = pd.read_csv(MONASH_BITCOIN_SAMPLE)
        values = frame.loc[frame["label"] == "price", "value"].astype(float).head(30).tolist()
    else:
        values = [100.0 + index * 1.5 for index in range(30)]
    return {
        "series_id": "bitcoin_price",
        "values": values,
        "horizon": 5,
        "frequency": "daily",
        "seasonal_period": 1,
        "dataset_name": "monash_bitcoin_okhs_sample",
    }

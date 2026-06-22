from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from benchmark.v2.core import (
    DatasetSpec,
    ForecastingSeriesRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_M5_DIR = PROJECT_ROOT / "examples" / "data" / "m5"

M5_HORIZON = 28
M5_FREQUENCY = "daily"
M5_SEASONAL_PERIOD = 7


class BenchmarkConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class M5Hierarchy:

    item_id: str
    category_id: str
    department_id: str
    state_id: str
    store_id: str
    category_state: str
    department_state: str
    store: str

    @classmethod
    def from_item_id(cls, item_id: str) -> "M5Hierarchy":
        base = item_id.rsplit("_", 1)[0] if "_validation" in item_id or "_evaluation" in item_id else item_id
        parts = base.split("_")
        if len(parts) < 5:
            raise BenchmarkConfigurationError(...)

        category = parts[0]
        department_num = parts[1]
        item_num = parts[2]
        state = parts[3]
        store_num = parts[4]

        department_id = f"{category}_{department_num}"
        store_id = f"{state}_{store_num}"
        category_state = f"{category}_{state}"
        department_state = f"{department_id}_{state}"

        return cls(
            item_id=item_id,
            category_id=category,
            department_id=department_id,
            state_id=state,
            store_id=store_id,
            category_state=category_state,
            department_state=department_state,
            store=store_id,
        )


def _load_calendar(data_dir: Path) -> pd.DataFrame:
    """Load M5 calendar file."""
    calendar_path = data_dir / "calendar.csv"
    if not calendar_path.exists():
        raise BenchmarkConfigurationError(
            f"Calendar file not found: {calendar_path}. "
            "Please download M5 dataset from Kaggle."
        )
    calendar = pd.read_csv(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar


def _load_sell_prices(data_dir: Path) -> pd.DataFrame:
    """Load M5 sell prices file."""
    prices_path = data_dir / "sell_prices.csv"
    if not prices_path.exists():
        raise BenchmarkConfigurationError(f"Sell prices file not found: {prices_path}")
    return pd.read_csv(prices_path)


def _load_sales_data(data_dir: Path, sample_size: Optional[int] = None) -> tuple[pd.DataFrame, str]:
    """Load M5 sales data. Returns (dataframe, source)."""
    val_path = data_dir / "sales_train_validation.csv"
    eval_path = data_dir / "sales_train_evaluation.csv"

    if val_path.exists():
        sales = pd.read_csv(val_path)
        source = "validation"
    elif eval_path.exists():
        sales = pd.read_csv(eval_path)
        source = "evaluation"
    else:
        raise BenchmarkConfigurationError(
            f"No sales file found in {data_dir}. Expected sales_train_validation.csv "
            "or sales_train_evaluation.csv"
        )

    if sample_size is not None and sample_size > 0:
        sales = sales.head(sample_size)

    return sales, source


def _extract_series_values(sales_frame: pd.DataFrame, series_id: str) -> np.ndarray:
    """Extract time series values for a given series ID."""
    series_row = sales_frame[sales_frame["id"] == series_id]
    if series_row.empty:
        raise BenchmarkConfigurationError(f"Series ID not found: {series_id}")

    sales_cols = [col for col in sales_frame.columns if col.startswith("d_")]
    sales_cols_sorted = sorted(sales_cols, key=lambda x: int(x.split("_")[1]))

    return series_row[sales_cols_sorted].values.flatten().astype(float)


def _build_covariate_data(
    series_id: str,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    train_len: int,
    test_len: int,
) -> dict[str, Any]:
    """Build covariate data for the series (known future, static)."""
    hierarchy = M5Hierarchy.from_item_id(series_id)

    # --- Static covariates ---
    static_covariates = {
        "category_id": hierarchy.category_id,
        "department_id": hierarchy.department_id,
        "state_id": hierarchy.state_id,
        "store_id": hierarchy.store_id,
    }

    # --- Price known future (requires forecasting future prices — simplified) ---
    # In production, price forecasting is complex. Here we use last known price.
    item_prices = prices[(prices["item_id"] == hierarchy.item_id) & (prices["store_id"] == hierarchy.store_id)]
    if not item_prices.empty:
        last_price = float(item_prices["sell_price"].iloc[-1])
        known_future_price = np.full(test_len, last_price, dtype=float)
    else:
        known_future_price = np.ones(test_len, dtype=float)

    # --- Calendar known future (actually known in advance) ---
    # Get calendar dates for test period
    if len(calendar) >= train_len + test_len:
        test_calendar = calendar.iloc[train_len: train_len + test_len]
        known_future_calendar = {
            "weekday": test_calendar["wday"].values.astype(float),  # или int
            "event_1": test_calendar["event_name_1"].fillna("none").astype(str).values,
            "snap": test_calendar[[c for c in test_calendar.columns if c.startswith("snap_")]]
            .fillna(0).sum(axis=1).values.astype(float),
        }
    else:
        known_future_calendar = {
            "weekday": np.zeros(test_len, dtype=float),
            "event_1": np.full(test_len, "none", dtype=object),
            "snap": np.zeros(test_len, dtype=float),
        }

    return {
        "static_covariates": static_covariates,
        "known_future_covariates": {
            "price": known_future_price,
            **known_future_calendar,
        },
    }


def _split_series(
    sales_frame: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    series_id: str,
    horizon: int = M5_HORIZON,
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, Any]]:
    values = _extract_series_values(sales_frame, series_id)

    if len(values) <= horizon:
        raise BenchmarkConfigurationError(
            f"Series {series_id} has only {len(values)} days, need at least {horizon + 1}"
        )

    train_values = tuple(values[:-horizon].tolist())
    test_values = tuple(values[-horizon:].tolist())

    covariate_data = _build_covariate_data(
        series_id=series_id,
        calendar=calendar,
        prices=prices,
        train_len=len(train_values),
        test_len=horizon,
    )

    hierarchy = M5Hierarchy.from_item_id(series_id)

    metadata = {
        "split_provenance": "m5_fixed_holdout",
        "horizon": horizon,
        "train_days": len(train_values),
        "test_days": len(test_values),
        "hierarchy": {
            "item_id": hierarchy.item_id,
            "category_id": hierarchy.category_id,
            "department_id": hierarchy.department_id,
            "state_id": hierarchy.state_id,
            "store_id": hierarchy.store_id,
            "category_state": hierarchy.category_state,
            "department_state": hierarchy.department_state,
        },
        "static_covariates": covariate_data["static_covariates"],
        "known_future_covariates": covariate_data["known_future_covariates"],
    }

    return train_values, test_values, metadata


class M5Adapter:

    benchmark_name = "m5"

    def __init__(self, loader: Optional[Callable[[DatasetSpec], Any]] = None):
        self.loader = loader or self._default_loader

    def load_series(self, spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        """Load M5 series records."""
        data_dir = Path(spec.adapter_options.get("local_data_dir", DEFAULT_LOCAL_M5_DIR))
        include_calendar = spec.adapter_options.get("include_calendar", True)
        include_prices = spec.adapter_options.get("include_prices", True)

        records = self.loader(spec, data_dir, include_calendar, include_prices)
        return self._sample_records(records, spec)

    def _default_loader(
        self,
        spec: DatasetSpec,
        data_dir: Path,
        include_calendar: bool,
        include_prices: bool,
    ) -> tuple[ForecastingSeriesRecord, ...]:
        """Default M5 loader."""
        sales_frame, source = _load_sales_data(data_dir, spec.sample_size)
        calendar = _load_calendar(data_dir) if include_calendar else pd.DataFrame()
        prices = _load_sell_prices(data_dir) if include_prices else pd.DataFrame()

        series_ids = list(spec.series_ids) if spec.series_ids else sales_frame["id"].tolist()

        records: list[ForecastingSeriesRecord] = []

        for series_id in series_ids:
            try:
                train_values, test_values, metadata = _split_series(
                    sales_frame=sales_frame,
                    calendar=calendar,
                    prices=prices,
                    series_id=series_id,
                    horizon=M5_HORIZON,
                )

                metadata["source_file"] = source

                record = ForecastingSeriesRecord(
                    benchmark=self.benchmark_name,
                    dataset_name=spec.dataset_name,
                    subset=spec.subset,
                    series_id=series_id,
                    frequency=M5_FREQUENCY,
                    forecast_horizon=M5_HORIZON,
                    seasonal_period=M5_SEASONAL_PERIOD,
                    train_values=train_values,
                    test_values=test_values,
                    metadata=metadata,
                )
                records.append(record)

            except Exception as e:
                import warnings

                warnings.warn(f"Failed to load series {series_id}: {e}", stacklevel=2)
                continue

        if not records:
            raise BenchmarkConfigurationError(f"No valid M5 series were loaded from {data_dir}")

        return tuple(records)

    def _sample_records(
        self,
        records: tuple[ForecastingSeriesRecord, ...],
        spec: DatasetSpec,
    ) -> tuple[ForecastingSeriesRecord, ...]:
        filtered = list(records)

        if spec.series_ids:
            requested = set(spec.series_ids)
            filtered = [r for r in filtered if r.series_id in requested]

        if spec.sample_size is not None and len(filtered) > spec.sample_size:
            rng = np.random.default_rng(spec.random_seed)
            indices = rng.choice(len(filtered), size=spec.sample_size, replace=False)
            filtered = [filtered[i] for i in sorted(indices)]

        return tuple(filtered)

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Literal

import numpy as np
import pandas as pd

from benchmark.v2.core import (
    DatasetSpec,
    ForecastingDatasetRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MULTIVARIATE_DIR = PROJECT_ROOT / "examples" / "data" / "multivariate"


class BenchmarkConfigurationError(ValueError):
    pass


DATASET_CONFIGS = {
    "etth1": {
        "frequency": "hourly",
        "seasonal_period": 24,
        "n_features": 7,
        "feature_names": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "target_index": 6,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Electricity Transformer Temperature - Hourly 1",
    },
    "etth2": {
        "frequency": "hourly",
        "seasonal_period": 24,
        "n_features": 7,
        "feature_names": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "target_index": 6,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Electricity Transformer Temperature - Hourly 2",
    },
    "ettm1": {
        "frequency": "15min",
        "seasonal_period": 96,
        "n_features": 7,
        "feature_names": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "target_index": 6,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Electricity Transformer Temperature - 15min 1",
    },
    "ettm2": {
        "frequency": "15min",
        "seasonal_period": 96,
        "n_features": 7,
        "feature_names": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "target_index": 6,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Electricity Transformer Temperature - 15min 2",
    },
    "electricity": {
        "frequency": "hourly",
        "seasonal_period": 24,
        "n_features": 321,
        "feature_names": None,
        "target_index": None,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Electricity Consuming Load - 321 clients",
    },
    "exchange_rate": {
        "frequency": "daily",
        "seasonal_period": 7,
        "n_features": 8,
        "feature_names": None,
        "target_index": None,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Exchange Rate - 8 currencies",
    },
    "traffic": {
        "frequency": "hourly",
        "seasonal_period": 24,
        "n_features": 862,
        "feature_names": None,
        "target_index": None,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Traffic - 862 sensors",
    },
    "weather": {
        "frequency": "10min",
        "seasonal_period": 144,
        "n_features": 21,
        "feature_names": None,
        "target_index": None,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Weather - 21 indicators",
    },
    "solar": {
        "frequency": "10min",
        "seasonal_period": 144,
        "n_features": 137,
        "feature_names": None,
        "target_index": None,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "Solar Energy - 137 PV plants",
    },
    "pems03": {
        "frequency": "5min",
        "seasonal_period": 288,
        "n_features": 358,
        "feature_names": None,
        "target_index": None,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "PEMS03 Traffic Flow - 358 sensors",
        "special_loader": "pems_npy",
    },
    "pems04": {
        "frequency": "5min",
        "seasonal_period": 288,
        "n_features": 307,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "description": "PEMS04 Traffic Flow - 307 sensors",
    },
    "pems07": {
        "frequency": "5min",
        "seasonal_period": 288,
        "n_features": 883,
        "description": "PEMS07 Traffic Flow - 883 sensors",
    },
    "pems08": {
        "frequency": "5min",
        "seasonal_period": 288,
        "n_features": 170,
        "description": "PEMS08 Traffic Flow - 170 sensors",
    },
}


@dataclass
class MultivariateTimeSeries:
    data: np.ndarray
    feature_names: list[str]
    frequency: str
    seasonal_period: int

    def __post_init__(self):
        if self.data.ndim != 2:
            raise ValueError(f"Expected 2D array (T, C), got shape {self.data.shape}")

    @property
    def n_timesteps(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    def split_by_date(
            self,
            train_ratio: float = 0.6,
            val_ratio: float = 0.2,
            test_ratio: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split by time (canonical for LTSF literature)."""
        total = self.n_timesteps
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train = self.data[:train_end]
        val = self.data[train_end:val_end]
        test = self.data[val_end:]

        return train, val, test

    def split_by_window(
            self,
            context_len: int = 336,
            horizon: int = 96,
            stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create rolling windows for evaluation.

        Returns:
            X: shape (n_windows, context_len, C)
            y: shape (n_windows, horizon, C)
        """
        X, y = [], []
        total_len = self.n_timesteps

        for start in range(0, total_len - context_len - horizon + 1, stride):
            end_context = start + context_len
            end_forecast = end_context + horizon

            X.append(self.data[start:end_context])
            y.append(self.data[end_context:end_forecast])

        return np.array(X), np.array(y)


class MultivariateDatasetLoader:
    """
    Loader for multivariate time series datasets.

    Returns data in shape [T, C] - ready for PatchTST, iTransformer, etc.
    """

    benchmark_name = "multivariate"

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else DEFAULT_LOCAL_MULTIVARIATE_DIR

    def load(self, dataset_name: str, **kwargs) -> MultivariateTimeSeries:
        """Load dataset as MultivariateTimeSeries."""
        dataset_name = dataset_name.lower()

        if dataset_name not in DATASET_CONFIGS:
            raise BenchmarkConfigurationError(f"Unknown dataset: {dataset_name}")

        config = DATASET_CONFIGS[dataset_name]

        # Handle special loaders (e.g., PEMS .npz)
        if config.get("special_loader") == "pems_npy":
            return self._load_pems_npy(dataset_name, config)

        # Standard CSV loader
        file_path = self._find_file(dataset_name, config)
        return self._load_csv(file_path, dataset_name, config)

    def _find_file(self, dataset_name: str, config: dict) -> Path:
        """Find dataset file in data root."""
        # Try exact match
        candidates = [
            self.data_root / f"{dataset_name}.csv",
            self.data_root / dataset_name / f"{dataset_name}.csv",
            self.data_root / config.get("filename", f"{dataset_name}.csv"),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise BenchmarkConfigurationError(
            f"Dataset {dataset_name} not found in {self.data_root}. "
            f"Expected one of: {candidates}"
        )

    def _load_csv(self, file_path: Path, dataset_name: str, config: dict) -> MultivariateTimeSeries:
        """Load CSV as multivariate time series."""
        df = pd.read_csv(file_path)

        # Handle date column
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
            df = df.drop(columns=["date"])

        # Select features
        if config.get("feature_names"):
            feature_cols = [col for col in config["feature_names"] if col in df.columns]
        else:
            # Use all numeric columns
            feature_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        data = df[feature_cols].values.astype(np.float32)

        return MultivariateTimeSeries(
            data=data,
            feature_names=feature_cols,
            frequency=config["frequency"],
            seasonal_period=config["seasonal_period"],
        )

    def _load_pems_npy(self, dataset_name: str, config: dict) -> MultivariateTimeSeries:
        """Load PEMS dataset from .npz or .npy format."""
        import scipy.io as sio

        candidates = [
            self.data_root / f"{dataset_name}.npz",
            self.data_root / f"{dataset_name}.npy",
            self.data_root / dataset_name / f"{dataset_name}.npz",
        ]

        file_path = None
        for candidate in candidates:
            if candidate.exists():
                file_path = candidate
                break

        if file_path is None:
            raise BenchmarkConfigurationError(
                f"PEMS dataset {dataset_name} not found. Expected .npz or .npy file."
            )

        if file_path.suffix == ".npz":
            data = np.load(file_path)["data"]
        else:
            data = np.load(file_path)

        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        elif data.ndim == 1:
            data = data.reshape(-1, 1)

        return MultivariateTimeSeries(
            data=data.astype(np.float32),
            feature_names=[f"sensor_{i}" for i in range(data.shape[1])],
            frequency=config["frequency"],
            seasonal_period=config["seasonal_period"],
        )


class UnivariateExtractor:
    def __init__(self, multivariate: MultivariateTimeSeries):
        self.multivariate = multivariate

    def get_series_records(
        self,
        forecast_horizon: int,
        leakage_status: str | None = None,
        dataset_name: str | None = None,
    ) -> list[ForecastingDatasetRecord]:
        records = []
        for idx, feature_name in enumerate(self.multivariate.feature_names):
            values = self.multivariate.data[:, idx]
            if len(values) <= forecast_horizon:
                continue

            train_values = tuple(values[:-forecast_horizon].tolist())
            test_values = tuple(values[-forecast_horizon:].tolist())

            record = ForecastingDatasetRecord(
                benchmark="multivariate",
                dataset_name=dataset_name or self.multivariate.frequency,
                subset="default",
                series_id=feature_name,
                frequency=self.multivariate.frequency,
                forecast_horizon=forecast_horizon,
                seasonal_period=self.multivariate.seasonal_period,
                train_values=train_values,
                test_values=test_values,
                known_future_covariates={},
                observed_past_covariates={},
                static_covariates={
                    "feature_index": idx,
                    "total_features": self.multivariate.n_features,
                },
                panel_ids={},
                hierarchy={},
                metadata={
                    "source_type": "univariate_extraction",
                    "parent_dataset": self.multivariate.frequency,
                    "feature_index": idx,
                    "n_features_total": self.multivariate.n_features,
                },
                leakage_status=leakage_status,
            )
            records.append(record)
        return records

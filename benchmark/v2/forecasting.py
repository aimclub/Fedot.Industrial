from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional in lightweight test envs
    import torch
except Exception:  # pragma: no cover
    torch = None

from fedot_ind.core.models.kernel.okhs_common import OKHSMethod, QPolicy, canonical_method_name, normalize_okhs_method
from fedot_ind.core.models.ts_forecasting.dmd_models.havok_forecaster import HAVOKForecaster
from fedot_ind.core.models.ts_forecasting.lagged_model.mssa_forecaster import MSSAForecaster
from fedot_ind.core.models.ts_forecasting.progress_policy import (
    ForecastingProgressPolicy,
    resolve_forecasting_progress_policy,
)
from fedot_ind.core.models.ts_forecasting.regime_utils.regime_diagnostics import analyze_regime_diagnostics
from fedot_ind.core.models.ts_forecasting.regime_routing import adapter_name_to_family, recommend_forecasting_model
from fedot_ind.core.repository.forecasting_registry import canonical_forecasting_model_name

try:  # pragma: no cover - operator-model path may require torch/scipy-heavy stack
    from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
except Exception:  # pragma: no cover
    OKHSForecaster = None
try:  # pragma: no cover - DMD runtime may require torch stack
    from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd_forecasting import DMDForecaster
except Exception:  # pragma: no cover
    DMDForecaster = None
try:  # pragma: no cover - tensor-native composites require torch
    from fedot_ind.core.models.ts_forecasting.ensemble_models.hybrid_ensemble_forecaster import HybridEnsembleForecaster
    from fedot_ind.core.models.ts_forecasting.lagged_model.lagged_ridge_forecaster import LaggedRidgeForecaster
    from fedot_ind.core.models.ts_forecasting.lagged_model.low_rank_lagged_ridge_forecaster import \
        LowRankLaggedRidgeForecaster
    from fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head import \
        run_neural_forecast_head_on_series
    from fedot_ind.core.models.ts_forecasting.dmd_models.okhs_fdmd_forecaster import (
        OKHSFDMDForecaster,
        build_okhs_fdmd_spec,
        run_okhs_fdmd_forecaster_on_series,
    )
    from fedot_ind.core.models.ts_forecasting.forecasting_runtime import ForecastingSplitKind, ForecastingSplitSpec
    from fedot_ind.core.models.ts_forecasting.stage_tuning_runtime import run_forecasting_stage_tuning_on_series
except Exception:  # pragma: no cover
    HybridEnsembleForecaster = None
    LaggedRidgeForecaster = None
    LowRankLaggedRidgeForecaster = None
    run_neural_forecast_head_on_series = None
    OKHSFDMDForecaster = None
    build_okhs_fdmd_spec = None
    run_okhs_fdmd_forecaster_on_series = None
    ForecastingSplitKind = None
    ForecastingSplitSpec = None
    run_forecasting_stage_tuning_on_series = None

try:  # pragma: no cover - optional heavyweight dependency tree in test envs
    from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_LENGTH, M4_PREFIX, M4_SEASONALITY
except Exception:  # pragma: no cover - lightweight fallback for benchmark-v2
    M4_FORECASTING_LENGTH = {'D': 14, 'W': 13, 'M': 18, 'Q': 8, 'Y': 6}
    M4_SEASONALITY = {'D': 1, 'W': 1, 'M': 12, 'Q': 4, 'Y': 1}
    M4_PREFIX = {'D': 'Daily', 'W': 'Weekly',
                 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}

from benchmark.v2.core import (
    BenchmarkAggregateReport,
    BenchmarkRunRecord,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ForecastingBenchmarkResult,
    ForecastingModelAdapter,
    ForecastingSeriesRecord,
    MetricRecord,
    ModelSpec,
    PredictionRecord,
    RunStatus,
    TaskType,
    new_run_id,
    RunMode,
    ModelFamily,
)

from .forecasting_result import (
    coerce_forecast_result,
    describe_forecast_result_kind,
    iter_quantile_prediction_records,
    resolve_point_forecast,
    validate_forecast_result_shapes,
)

from benchmark.v2.verbosity import (
    ForecastingVerbosityPolicy,
    resolve_forecasting_verbosity_policy,
)
from benchmark.v2.progress import BenchmarkProgressMonitor
from benchmark.v2.incremental_persistence import ForecastingIncrementalPersistenceCoordinator

SUPPORTED_FORECASTING_METRICS = ('mase', 'smape', 'owa', 'rmse', 'mae')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_M4_DIR = PROJECT_ROOT / 'examples' / 'data' / 'm4' / 'datasets'
DEFAULT_LOCAL_MONASH_DIR = PROJECT_ROOT / 'examples' / \
    'data' / 'benchmark' / 'forecasting' / 'monash_benchmark'


class BenchmarkConfigurationError(ValueError):
    """Raised when a forecasting benchmark configuration is invalid."""


class ModelExecutionError(RuntimeError):
    """Raised when a model run produces a benchmark-level failure status."""

    def __init__(self, status: RunStatus, message: str):
        """Store the run status alongside the human-readable error message."""
        super().__init__(message)
        self.status = status
        self.message = message


def validate_forecasting_suite_config(config: BenchmarkSuiteConfig) -> None:
    """Validate that a suite config can be executed as a forecasting benchmark."""
    if config.task_type is not TaskType.FORECASTING:
        raise BenchmarkConfigurationError(
            'Forecasting suite expects task_type=forecasting.')
    if not config.datasets:
        raise BenchmarkConfigurationError(
            'Benchmark suite must contain at least one dataset spec.')
    if not config.models:
        raise BenchmarkConfigurationError(
            'Benchmark suite must contain at least one model spec.')
    unsupported = set(config.metrics) - set(SUPPORTED_FORECASTING_METRICS)
    if unsupported:
        raise BenchmarkConfigurationError(
            f'Unsupported forecasting metrics: {sorted(unsupported)}')


def _normalize_subset_name(subset: str) -> str:
    mapping = {
        'd': 'Daily',
        'daily': 'Daily',
        'w': 'Weekly',
        'weekly': 'Weekly',
        'm': 'Monthly',
        'monthly': 'Monthly',
        'q': 'Quarterly',
        'quarterly': 'Quarterly',
        'y': 'Yearly',
        'yearly': 'Yearly',
    }
    return mapping.get(subset.lower(), subset)


def _infer_m4_key(subset: str) -> str:
    normalized = _normalize_subset_name(subset)
    reverse = {value: key for key, value in M4_PREFIX.items()}
    if normalized not in reverse:
        raise BenchmarkConfigurationError(f'Unsupported M4 subset: {subset}')
    return reverse[normalized]


def _sample_records(
        records: list[ForecastingSeriesRecord],
        spec: DatasetSpec,
) -> tuple[ForecastingSeriesRecord, ...]:
    filtered = records
    if spec.series_ids:
        requested = set(spec.series_ids)
        filtered = [
            record for record in filtered if record.series_id in requested]
    if spec.sample_size is not None and len(filtered) > spec.sample_size:
        rng = np.random.default_rng(spec.random_seed)
        indices = rng.choice(
            len(filtered), size=spec.sample_size, replace=False)
        filtered = [filtered[index] for index in sorted(indices)]
    return tuple(filtered)


def _resolve_stage_tuning_split_spec(raw: dict[str, Any] | None):
    if not raw or ForecastingSplitSpec is None:
        return None
    split_spec = raw.get('split_spec')
    if isinstance(split_spec, ForecastingSplitSpec):
        return split_spec
    if isinstance(split_spec, dict):
        kind = split_spec.get('kind')
        validation_horizon = split_spec.get('validation_horizon')
        min_train_length = split_spec.get('min_train_length')
        n_splits = split_spec.get('n_splits')
        test_size = split_spec.get('test_size')
        gap = split_spec.get('gap', 0)
        max_train_size = split_spec.get('max_train_size')
        initial_window = split_spec.get('initial_window')
        step_length = split_spec.get('step_length')
        if kind is not None and ForecastingSplitKind is not None:
            kind = ForecastingSplitKind(str(kind).lower())
        return ForecastingSplitSpec(
            kind=kind or ForecastingSplitSpec().kind,
            validation_horizon=validation_horizon,
            min_train_length=min_train_length,
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size,
            initial_window=initial_window,
            step_length=step_length,
        )
    validation_horizon = raw.get('validation_horizon')
    min_train_length = raw.get('min_train_length')
    n_splits = raw.get('n_splits')
    test_size = raw.get('test_size')
    gap = raw.get('gap')
    max_train_size = raw.get('max_train_size')
    initial_window = raw.get('initial_window')
    step_length = raw.get('step_length')
    if (
            validation_horizon is None
            and min_train_length is None
            and n_splits is None
            and test_size is None
            and gap is None
            and max_train_size is None
            and initial_window is None
            and step_length is None
    ):
        return None
    return ForecastingSplitSpec(
        validation_horizon=validation_horizon,
        min_train_length=min_train_length,
        n_splits=n_splits,
        test_size=test_size,
        gap=0 if gap is None else gap,
        max_train_size=max_train_size,
        initial_window=initial_window,
        step_length=step_length,
    )


def _maybe_attach_stage_tuning_report(
        metadata: dict[str, Any],
        *,
        adapter_name: str,
        series_record: ForecastingSeriesRecord,
        base_params: dict[str, Any],
        runtime_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not runtime_config:
        return metadata
    if run_forecasting_stage_tuning_on_series is None:
        metadata['stage_tuning_report_error'] = 'stage_tuning_runtime is unavailable in the current environment.'
        return metadata

    config = dict(runtime_config)
    metric_name = str(config.get('metric_name', 'rmse'))
    stage_updates = config.get('stage_updates')
    max_values_per_parameter = int(config.get('max_values_per_parameter', 3))
    max_stage_candidates = int(config.get('max_stage_candidates', 16))
    split_spec = _resolve_stage_tuning_split_spec(config)
    progress_policy = config.get('progress_policy')
    verbosity_policy = config.get('verbosity_policy')
    resolved_verbosity_policy = (
        verbosity_policy
        if isinstance(verbosity_policy, ForecastingVerbosityPolicy)
        else resolve_forecasting_verbosity_policy(
            (verbosity_policy or {}).get('level') if isinstance(
                verbosity_policy, dict) else verbosity_policy,
            options=verbosity_policy if isinstance(
                verbosity_policy, dict) else None,
        )
    )

    try:
        report = run_forecasting_stage_tuning_on_series(
            adapter_name,
            time_series=np.asarray(series_record.train_values, dtype=float),
            forecast_horizon=series_record.forecast_horizon,
            base_params=base_params,
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=series_record.seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
            progress_policy=progress_policy,
        )
        stage_tuning_report = resolved_verbosity_policy.prune_stage_tuning_report(
            report.to_dict())
        if stage_tuning_report is not None:
            metadata['stage_tuning_report'] = stage_tuning_report
        stage_tuning_runtime = {
            'enabled': True,
            'metric_name': metric_name,
            'max_values_per_parameter': max_values_per_parameter,
            'max_stage_candidates': max_stage_candidates,
            'improved': report.metadata.get('improved'),
            'baseline_score': report.metadata.get('baseline_score'),
            'best_score': report.metadata.get('best_score'),
            'progress_policy': resolve_forecasting_progress_policy(progress_policy).to_dict(),
        }
        pruned_runtime = resolved_verbosity_policy.prune_stage_tuning_runtime(
            stage_tuning_runtime)
        if pruned_runtime is not None:
            metadata['stage_tuning_runtime'] = pruned_runtime
    except Exception as exc:  # pragma: no cover - benchmark should keep the main run alive
        metadata['stage_tuning_report_error'] = str(exc)
    return metadata


def _series_split_from_full_values(
        values: np.ndarray,
        forecast_horizon: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    normalized = np.asarray(values, dtype=float).reshape(-1)
    if len(normalized) <= forecast_horizon:
        raise BenchmarkConfigurationError(
            f'Series length {len(normalized)} must be greater than forecast horizon {forecast_horizon}.'
        )
    train = tuple(float(value) for value in normalized[:-forecast_horizon])
    test = tuple(float(value) for value in normalized[-forecast_horizon:])
    return train, test


def _parse_frame_records(
        frame: pd.DataFrame,
        *,
        benchmark: str,
        dataset_name: str,
        subset: str,
        default_frequency: str,
        default_horizon: int,
        default_seasonal_period: int,
) -> list[ForecastingSeriesRecord]:
    identifier_column = 'series_id'
    for candidate in ('series_id', 'unique_id', 'item_id'):
        if candidate in frame.columns:
            identifier_column = candidate
            break
    if identifier_column not in frame.columns:
        raise BenchmarkConfigurationError(
            'Dataset frame must include series_id or unique_id column.')

    value_column = 'value'
    for candidate in ('value', 'y', 'target'):
        if candidate in frame.columns:
            value_column = candidate
            break
    if value_column not in frame.columns:
        raise BenchmarkConfigurationError(
            'Dataset frame must include value/y/target column.')

    sort_columns = [candidate for candidate in (
        'timestamp', 'datetime', 'ds') if candidate in frame.columns]
    records: list[ForecastingSeriesRecord] = []
    for series_id, group in frame.groupby(identifier_column):
        ordered = group.sort_values(sort_columns) if sort_columns else group
        series_values = ordered[value_column].astype(float).to_numpy()
        horizon = int(ordered['horizon'].iloc[0]
                      ) if 'horizon' in ordered.columns else default_horizon
        seasonal_period = (
            int(ordered['seasonal_period'].iloc[0])
            if 'seasonal_period' in ordered.columns
            else default_seasonal_period
        )
        frequency = str(ordered['frequency'].iloc[0]
                        ) if 'frequency' in ordered.columns else default_frequency
        record_dataset_name = (
            str(ordered['dataset_name'].iloc[0]
                ) if 'dataset_name' in ordered.columns else dataset_name
        )
        train_values, test_values = _series_split_from_full_values(
            series_values, horizon)
        records.append(
            ForecastingSeriesRecord(
                benchmark=benchmark,
                dataset_name=record_dataset_name,
                subset=subset,
                series_id=str(series_id),
                frequency=frequency,
                forecast_horizon=horizon,
                seasonal_period=seasonal_period,
                train_values=train_values,
                test_values=test_values,
                metadata={'split_provenance': 'tail_holdout_from_full_series'},
            )
        )
    return records


def _parse_sequence_records(
        payload: list[dict[str, Any]],
        *,
        benchmark: str,
        dataset_name: str,
        subset: str,
        default_frequency: str,
        default_horizon: int,
        default_seasonal_period: int,
) -> list[ForecastingSeriesRecord]:
    records: list[ForecastingSeriesRecord] = []
    for index, item in enumerate(payload):
        series_id = str(item.get('series_id', item.get(
            'unique_id', f'{dataset_name}_{index}')))
        frequency = str(item.get('frequency', default_frequency))
        horizon = int(item.get('horizon', default_horizon))
        seasonal_period = int(
            item.get('seasonal_period', default_seasonal_period))
        values = item.get('values')
        train = item.get('train_values')
        test = item.get('test_values')
        if values is not None:
            train_values, test_values = _series_split_from_full_values(
                np.asarray(values, dtype=float), horizon)
        elif train is not None and test is not None:
            train_values = tuple(float(value) for value in np.asarray(
                train, dtype=float).reshape(-1))
            test_values = tuple(float(value) for value in np.asarray(
                test, dtype=float).reshape(-1))
        else:
            raise BenchmarkConfigurationError(
                'Sequence payload must include values or train_values/test_values.')

        records.append(
            ForecastingSeriesRecord(
                benchmark=benchmark,
                dataset_name=str(item.get('dataset_name', dataset_name)),
                subset=subset,
                series_id=series_id,
                frequency=frequency,
                forecast_horizon=horizon,
                seasonal_period=seasonal_period,
                train_values=train_values,
                test_values=test_values,
                metadata={'split_provenance': item.get(
                    'split_provenance', 'adapter_provided')},
            )
        )
    return records


class M4Adapter:
    """Dataset adapter for loading M4 forecasting series."""

    benchmark_name = 'm4'

    def __init__(self, loader: Callable[[DatasetSpec], Any] | None = None):
        """Store a custom or default M4 loader."""
        self.loader = loader or self._default_loader

    def load_series(self, spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        """Load, normalize and optionally sample M4 series records."""
        subset = _normalize_subset_name(spec.subset)
        key = _infer_m4_key(subset)
        local_csv_dir = spec.adapter_options.get('local_csv_dir')
        use_local_files = spec.adapter_options.get('use_local_files', False)
        if local_csv_dir or use_local_files:
            records = self._load_local_records(spec, subset, key)
        else:
            frame = self.loader(spec)
            records = _parse_frame_records(
                frame=frame,
                benchmark=self.benchmark_name,
                dataset_name=spec.dataset_name,
                subset=subset,
                default_frequency=subset,
                default_horizon=M4_FORECASTING_LENGTH[key],
                default_seasonal_period=M4_SEASONALITY[key],
            )
        return _sample_records(records, spec)

    def _load_local_records(
            self,
            spec: DatasetSpec,
            subset: str,
            key: str,
    ) -> list[ForecastingSeriesRecord]:
        base_dir = Path(spec.adapter_options.get(
            'local_csv_dir', DEFAULT_LOCAL_M4_DIR))
        train_path = base_dir / f'{subset}-train.csv'
        test_path = base_dir / f'{subset}-test.csv'
        if not train_path.exists() or not test_path.exists():
            raise BenchmarkConfigurationError(
                f'Local M4 files were not found: {train_path} and {test_path}.'
            )

        train_frame = pd.read_csv(train_path)
        test_frame = pd.read_csv(test_path)
        if 'V1' not in train_frame.columns or 'V1' not in test_frame.columns:
            raise BenchmarkConfigurationError(
                'Local M4 CSV files must include the V1 identifier column.')

        train_frame = train_frame.set_index('V1')
        test_frame = test_frame.set_index('V1')
        common_ids = [
            series_id for series_id in train_frame.index if series_id in test_frame.index]
        if not common_ids:
            raise BenchmarkConfigurationError(
                'Local M4 train/test files do not share any series identifiers.')

        records: list[ForecastingSeriesRecord] = []
        for series_id in common_ids:
            train_values = train_frame.loc[series_id].dropna().astype(
                float).to_numpy()
            test_values = test_frame.loc[series_id].dropna().astype(
                float).to_numpy()
            if len(test_values) == 0 or len(train_values) == 0:
                continue
            records.append(
                ForecastingSeriesRecord(
                    benchmark=self.benchmark_name,
                    dataset_name=spec.dataset_name or subset,
                    subset=subset,
                    series_id=str(series_id),
                    frequency=subset,
                    forecast_horizon=int(len(test_values)),
                    seasonal_period=M4_SEASONALITY[key],
                    train_values=tuple(float(value) for value in train_values),
                    test_values=tuple(float(value) for value in test_values),
                    metadata={'split_provenance': 'local_m4_train_test_csv'},
                )
            )
        return records

    @staticmethod
    def _default_loader(spec: DatasetSpec) -> pd.DataFrame:
        try:
            from datasetsforecast.m4 import M4
        except Exception as exc:  # pragma: no cover
            raise BenchmarkConfigurationError(
                f'M4 adapter is unavailable: {exc}') from exc

        subset = _normalize_subset_name(spec.subset)
        data_directory = spec.adapter_options.get('data_dir')
        frame, _, _ = M4.load(directory=data_directory, group=subset)
        return frame


class MonashAdapter:
    """Dataset adapter for loading Monash-style forecasting series."""

    benchmark_name = 'monash'

    def __init__(self, loader: Callable[[DatasetSpec], Any] | None = None):
        """Store a custom or default Monash loader."""
        self.loader = loader or self._default_loader

    def load_series(self, spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        """Load, normalize and optionally sample Monash series records."""
        local_csv_path = spec.adapter_options.get('local_csv_path')
        use_local_files = spec.adapter_options.get('use_local_files', False)
        if local_csv_path or use_local_files:
            records = self._load_local_records(spec)
        else:
            payload = self.loader(spec)
            if isinstance(payload, pd.DataFrame):
                records = _parse_frame_records(
                    frame=payload,
                    benchmark=self.benchmark_name,
                    dataset_name=spec.dataset_name,
                    subset=spec.subset,
                    default_frequency=str(
                        spec.adapter_options.get('frequency', spec.subset)),
                    default_horizon=int(
                        spec.adapter_options.get('forecast_horizon', 1)),
                    default_seasonal_period=int(
                        spec.adapter_options.get('seasonal_period', 1)),
                )
            else:
                records = _parse_sequence_records(
                    payload=list(payload),
                    benchmark=self.benchmark_name,
                    dataset_name=spec.dataset_name,
                    subset=spec.subset,
                    default_frequency=str(
                        spec.adapter_options.get('frequency', spec.subset)),
                    default_horizon=int(
                        spec.adapter_options.get('forecast_horizon', 1)),
                    default_seasonal_period=int(
                        spec.adapter_options.get('seasonal_period', 1)),
                )
        return _sample_records(records, spec)

    def _load_local_records(self, spec: DatasetSpec) -> list[ForecastingSeriesRecord]:
        local_path = spec.adapter_options.get('local_csv_path')
        if local_path is None:
            file_candidates = sorted(
                DEFAULT_LOCAL_MONASH_DIR.glob(f'*{spec.dataset_name}*.csv'))
            if not file_candidates:
                raise BenchmarkConfigurationError(
                    f'No local Monash file matching dataset_name={spec.dataset_name} was found in {DEFAULT_LOCAL_MONASH_DIR}.'
                )
            local_path = file_candidates[0]
        csv_path = Path(local_path)
        if not csv_path.exists():
            raise BenchmarkConfigurationError(
                f'Local Monash CSV file was not found: {csv_path}')

        frame = pd.read_csv(csv_path)
        if 'label' not in frame.columns or 'value' not in frame.columns:
            raise BenchmarkConfigurationError(
                'Local Monash CSV must include `label` and `value` columns.')

        frequency = str(spec.adapter_options.get('frequency', spec.subset))
        horizon = int(spec.adapter_options.get('forecast_horizon',
                      self._infer_horizon_from_filename(csv_path)))
        seasonal_period = int(spec.adapter_options.get(
            'seasonal_period', self._infer_seasonal_period(frequency)))
        dataset_name = spec.dataset_name or csv_path.stem

        records: list[ForecastingSeriesRecord] = []
        sort_columns = [column for column in (
            'datetime', 'timestamp', 'ds') if column in frame.columns]
        for series_id, group in frame.groupby('label'):
            ordered = group.sort_values(
                sort_columns) if sort_columns else group
            values = ordered['value'].astype(float).to_numpy()
            if len(values) <= horizon:
                continue
            train_values, test_values = _series_split_from_full_values(
                values, horizon)
            records.append(
                ForecastingSeriesRecord(
                    benchmark=self.benchmark_name,
                    dataset_name=dataset_name,
                    subset=spec.subset,
                    series_id=str(series_id),
                    frequency=frequency,
                    forecast_horizon=horizon,
                    seasonal_period=seasonal_period,
                    train_values=train_values,
                    test_values=test_values,
                    metadata={
                        'split_provenance': 'local_monash_csv',
                        'source_file': csv_path.name,
                    },
                )
            )
        return records

    @staticmethod
    def _infer_horizon_from_filename(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.rsplit('_', maxsplit=1)[-1])
        except Exception:
            return 1

    @staticmethod
    def _infer_seasonal_period(frequency: str) -> int:
        normalized = frequency.lower()
        if 'month' in normalized:
            return 12
        if 'quarter' in normalized:
            return 4
        if 'week' in normalized:
            return 52
        if 'day' in normalized:
            return 7
        if 'hour' in normalized:
            return 24
        return 1

    @staticmethod
    def _default_loader(spec: DatasetSpec) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover
            raise BenchmarkConfigurationError(
                f'Monash adapter is unavailable: {exc}') from exc

        dataset_path = spec.adapter_options.get(
            'dataset_path', spec.dataset_name)
        split_name = spec.adapter_options.get('split', 'train')
        dataset = load_dataset(dataset_path, split=split_name)
        payload: list[dict[str, Any]] = []
        for row in dataset:
            payload.append(
                {
                    'series_id': row.get('item_id', row.get('series_id', row.get('unique_id'))),
                    'values': row.get('target', row.get('values')),
                    'horizon': row.get('horizon', spec.adapter_options.get('forecast_horizon', 1)),
                    'frequency': row.get('freq', row.get('frequency', spec.subset)),
                    'seasonal_period': row.get(
                        'seasonal_period',
                        spec.adapter_options.get('seasonal_period', 1),
                    ),
                    'dataset_name': spec.dataset_name,
                    'split_provenance': 'monash_dataset_loader',
                }
            )
        return payload


class InMemoryForecastingAdapter:
    """Dataset adapter backed by records provided directly in DatasetSpec."""

    benchmark_name = 'in_memory'

    def load_series(self, spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        """Convert in-memory payload records into ForecastingSeriesRecord values."""
        payload = list(spec.adapter_options.get('records', ()))
        if not payload:
            raise BenchmarkConfigurationError(
                'InMemory adapter requires adapter_options["records"].')
        records = _parse_sequence_records(
            payload=payload,
            benchmark=self.benchmark_name,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            default_frequency=str(
                spec.adapter_options.get('frequency', spec.subset)),
            default_horizon=int(
                spec.adapter_options.get('forecast_horizon', 1)),
            default_seasonal_period=int(
                spec.adapter_options.get('seasonal_period', 1)),
        )
        return _sample_records(records, spec)


def _safe_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _build_fedot_forecasting_input(series_record: ForecastingSeriesRecord):
    try:
        from fedot.core.data.data import InputData
        from fedot.core.repository.dataset_types import DataTypesEnum
        from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    except Exception as exc:  # pragma: no cover - depends on full FEDOT runtime
        raise ModelExecutionError(
            RunStatus.NOT_AVAILABLE, f'FEDOT forecasting runtime is unavailable: {exc}') from exc

    train = np.asarray(series_record.train_values, dtype=float).reshape(-1, 1)
    return InputData(
        idx=np.arange(len(train)),
        features=train,
        target=train.reshape(-1),
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(
            forecast_length=series_record.forecast_horizon)),
        data_type=DataTypesEnum.ts,
    )


@dataclass
class NaiveLastValueModel(ForecastingModelAdapter):
    """Simple baseline that repeats the last observed value."""

    name: str = 'NaiveLastValue'
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast by repeating the last train value."""
        train = np.asarray(series_record.train_values, dtype=float)
        return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'last_value'}


@dataclass
class NaiveMeanModel(ForecastingModelAdapter):
    """Simple baseline that repeats the train mean."""

    name: str = 'NaiveMean'
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast by repeating the mean train value."""
        train = np.asarray(series_record.train_values, dtype=float)
        return np.full(series_record.forecast_horizon, np.mean(train), dtype=float), {'strategy': 'mean'}


@dataclass
class NaiveDriftModel(ForecastingModelAdapter):
    """Simple baseline that extrapolates a linear end-to-end drift."""

    name: str = 'NaiveDrift'
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast a drift line from first to last train value."""
        train = np.asarray(series_record.train_values, dtype=float)
        if len(train) <= 1:
            return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'fallback'}
        slope = (train[-1] - train[0]) / max(len(train) - 1, 1)
        horizon = np.arange(1, series_record.forecast_horizon + 1, dtype=float)
        return train[-1] + slope * horizon, {'strategy': 'drift'}


@dataclass
class MovingAverageModel(ForecastingModelAdapter):
    """Simple baseline that repeats the trailing moving average."""

    window_size: int = 3
    name: str = 'MovingAverage'
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast by repeating the last-window mean."""
        train = np.asarray(series_record.train_values, dtype=float)
        window = min(max(self.window_size, 1), len(train))
        return np.full(series_record.forecast_horizon, np.mean(train[-window:]), dtype=float), {'window_size': window}


@dataclass
class LinearTrendModel(ForecastingModelAdapter):
    """Simple baseline that extrapolates a fitted linear trend."""

    name: str = 'LinearTrend'
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit a linear trend on train values and extrapolate the horizon."""
        train = np.asarray(series_record.train_values, dtype=float)
        if len(train) <= 1:
            return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'constant'}
        index = np.arange(len(train), dtype=float)
        slope, intercept = np.polyfit(index, train, deg=1)
        future_index = np.arange(len(train), len(
            train) + series_record.forecast_horizon, dtype=float)
        return intercept + slope * future_index, {'slope': float(slope), 'intercept': float(intercept)}


@dataclass
class ClassicalDMDModel(ForecastingModelAdapter):
    """Benchmark adapter for the classical DMD forecasting backend."""

    window_size: int = 12
    n_modes: int | None = None
    name: str = 'ClassicalDMD'
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'dmd')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether the DMD runtime dependencies are available."""
        if DMDForecaster is None or not _safe_import('torch'):
            return RunStatus.NOT_AVAILABLE, 'torch is required for DMDForecaster.'
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit DMD on Hankel trajectories and return the forecast horizon."""
        train = np.asarray(series_record.train_values, dtype=float)
        window_size = min(
            max(self.window_size, series_record.forecast_horizon + 1), len(train) - 1)
        if window_size <= series_record.forecast_horizon:
            raise ModelExecutionError(
                RunStatus.SKIPPED, 'Series is too short for ClassicalDMD windowing.')
        trajectories = np.array(
            [train[index:index + window_size]
                for index in range(len(train) - window_size + 1)],
            dtype=float,
        )
        model = DMDForecaster(
            forecast_horizon=series_record.forecast_horizon,
            n_modes=self.n_modes,
            use_koopman=False,
            epochs=1,
            device='cpu',
        )
        model.fit(trajectories, window_size=window_size)
        prediction = np.asarray(model.predict(
            train[-window_size:]), dtype=float).reshape(-1)
        return prediction[:series_record.forecast_horizon], {'window_size': window_size}


@dataclass
class OKHSModel(ForecastingModelAdapter):
    """Benchmark adapter for the legacy OKHS forecasting backend."""

    method: str | OKHSMethod = OKHSMethod.DMD
    q: float = 0.7
    q_policy: str | QPolicy = QPolicy.FIXED
    n_modes: int = 5
    window_size: int = 20
    window_policy: str = 'adaptive_cycle_aware'
    trajectory_representation_policy: str = 'projected'
    latent_trajectory_stride_policy: str = 'adaptive'
    latent_trajectory_stride: int | None = None
    mode_selection_policy: str = "energy"
    mode_energy_threshold: float = 0.95
    prediction_mode_selection_policy: str = "adaptive_tail_energy"
    max_prediction_modes: int | None = None
    min_prediction_modes: int = 4
    boundary_alignment_policy: str = "tapered_offset"
    boundary_alignment_decay: float = 4.0
    prediction_stability_threshold: float | None = 0.03
    anti_smoothing_policy: str = "residual_bridge"
    anti_smoothing_tail_window: int | None = None
    anti_smoothing_amplitude_ratio: float = 0.35
    anti_smoothing_monotone_ratio: float = 0.9
    anti_smoothing_oscillation_floor: float = 0.25
    anti_smoothing_decay: float = 2.5
    anti_smoothing_target_amplitude_ratio: float = 0.8
    name: str = 'OKHS'
    family: ModelFamily = ModelFamily.INTERNAL_INDUSTRIAL
    tags: tuple[str, ...] = ('okhs', 'forecasting')
    optional: bool = False
    device = 'cuda' if _safe_import(
        'torch') and torch.cuda.is_available() else 'cpu'

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether OKHS runtime dependencies are available."""
        if OKHSForecaster is None:
            return RunStatus.NOT_AVAILABLE, 'torch/runtime dependencies are required for OKHS forecasting.'
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit OKHS on train values and return forecast plus diagnostics."""
        train = np.asarray(series_record.train_values, dtype=float)
        window_size = min(max(self.window_size, 4), len(train) - 1)
        if window_size < 2:
            raise ModelExecutionError(
                RunStatus.SKIPPED, 'Series is too short for OKHS forecasting.')
        method = normalize_okhs_method(self.method)
        model = OKHSForecaster(
            q=self.q,
            forecast_horizon=series_record.forecast_horizon,
            n_modes=self.n_modes,
            method=method,
            q_policy=self.q_policy,
            window_policy=self.window_policy,
            trajectory_representation_policy=self.trajectory_representation_policy,
            latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
            latent_trajectory_stride=self.latent_trajectory_stride,
            mode_selection_policy=self.mode_selection_policy,
            mode_energy_threshold=self.mode_energy_threshold,
            prediction_mode_selection_policy=self.prediction_mode_selection_policy,
            max_prediction_modes=self.max_prediction_modes,
            min_prediction_modes=self.min_prediction_modes,
            boundary_alignment_policy=self.boundary_alignment_policy,
            boundary_alignment_decay=self.boundary_alignment_decay,
            prediction_stability_threshold=self.prediction_stability_threshold,
            anti_smoothing_policy=self.anti_smoothing_policy,
            anti_smoothing_tail_window=self.anti_smoothing_tail_window,
            anti_smoothing_amplitude_ratio=self.anti_smoothing_amplitude_ratio,
            anti_smoothing_monotone_ratio=self.anti_smoothing_monotone_ratio,
            anti_smoothing_oscillation_floor=self.anti_smoothing_oscillation_floor,
            anti_smoothing_decay=self.anti_smoothing_decay,
            anti_smoothing_target_amplitude_ratio=self.anti_smoothing_target_amplitude_ratio,
            device=self.device,
        )
        model.fit(train, window_size=window_size)
        model_prediction = model.predict(train)
        # Ensure prediction is on CPU before converting to numpy
        if hasattr(model_prediction, 'cpu'):
            model_prediction = model_prediction.cpu()
        prediction = np.asarray(model_prediction, dtype=float).reshape(-1)
        forecast = prediction[:series_record.forecast_horizon]
        metadata = {
            **model.get_optimization_info(),
            'selected_q': float(getattr(model, 'resolved_q_', self.q)),
            'method': canonical_method_name(method),
            'window_size': window_size,
            'last_train_value': float(train[-1]),
            'first_prediction_value': float(forecast[0]) if len(forecast) else None,
            'first_actual_value': float(series_record.test_values[0]) if series_record.test_values else None,
        }
        if metadata['first_prediction_value'] is not None:
            metadata['first_step_delta'] = float(
                metadata['first_prediction_value'] - metadata['last_train_value'])
        if metadata['first_actual_value'] is not None:
            metadata['first_actual_delta'] = float(
                metadata['first_actual_value'] - metadata['last_train_value'])
        return forecast, metadata


@dataclass
class OKHSFDMDForecasterModel(ForecastingModelAdapter):
    """Benchmark adapter for the named OKHS fDMD forecaster."""

    q: float = 0.7
    n_modes: int = 5
    window_size: int = 20
    q_policy: str = 'fixed'
    window_policy: str = 'adaptive_cycle_aware'
    trajectory_sampling_policy: str = 'dense'
    trajectory_rank_policy: str = 'explained_dispersion'
    trajectory_rank_value: int | None = None
    trajectory_representation_policy: str = 'projected'
    latent_trajectory_stride_policy: str = 'adaptive'
    latent_trajectory_stride: int | None = None
    mode_selection_policy: str = 'energy'
    mode_energy_threshold: float = 0.95
    prediction_mode_selection_policy: str = 'adaptive_tail_energy'
    max_prediction_modes: int | None = None
    min_prediction_modes: int = 4
    boundary_alignment_policy: str = 'tapered_offset'
    boundary_alignment_decay: float = 4.0
    prediction_stability_threshold: float | None = 0.03
    anti_smoothing_policy: str = 'residual_bridge'
    anti_smoothing_tail_window: int | None = None
    anti_smoothing_amplitude_ratio: float = 0.35
    anti_smoothing_monotone_ratio: float = 0.9
    anti_smoothing_oscillation_floor: float = 0.25
    anti_smoothing_decay: float = 2.5
    anti_smoothing_target_amplitude_ratio: float = 0.8
    stage_tuning_runtime: dict[str, Any] | None = None
    name: str = 'okhs_fdmd_forecaster'
    family: ModelFamily = ModelFamily.INTERNAL_INDUSTRIAL
    tags: tuple[str, ...] = ('okhs', 'operator_model', 'forecasting')
    optional: bool = False
    device = 'cuda' if _safe_import(
        'torch') and torch.cuda.is_available() else 'cpu'

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether the OKHS fDMD runtime is importable."""
        if (
                OKHSFDMDForecaster is None
                or build_okhs_fdmd_spec is None
                or run_okhs_fdmd_forecaster_on_series is None
        ):
            return RunStatus.NOT_AVAILABLE, 'torch/runtime dependencies are required for okhs_fdmd_forecaster.'
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Run OKHS fDMD on a benchmark series and attach stage diagnostics."""
        train = np.asarray(series_record.train_values, dtype=float)
        spec = build_okhs_fdmd_spec(
            forecast_horizon=series_record.forecast_horizon,
            params={
                'q': self.q,
                'n_modes': self.n_modes,
                'window_size': self.window_size,
                'q_policy': self.q_policy,
                'window_policy': self.window_policy,
                'trajectory_sampling_policy': self.trajectory_sampling_policy,
                'trajectory_rank_policy': self.trajectory_rank_policy,
                'trajectory_rank_value': self.trajectory_rank_value,
                'trajectory_representation_policy': self.trajectory_representation_policy,
                'latent_trajectory_stride_policy': self.latent_trajectory_stride_policy,
                'latent_trajectory_stride': self.latent_trajectory_stride,
                'mode_selection_policy': self.mode_selection_policy,
                'mode_energy_threshold': self.mode_energy_threshold,
                'prediction_mode_selection_policy': self.prediction_mode_selection_policy,
                'max_prediction_modes': self.max_prediction_modes,
                'min_prediction_modes': self.min_prediction_modes,
                'boundary_alignment_policy': self.boundary_alignment_policy,
                'boundary_alignment_decay': self.boundary_alignment_decay,
                'prediction_stability_threshold': self.prediction_stability_threshold,
                'anti_smoothing_policy': self.anti_smoothing_policy,
                'anti_smoothing_tail_window': self.anti_smoothing_tail_window,
                'anti_smoothing_amplitude_ratio': self.anti_smoothing_amplitude_ratio,
                'anti_smoothing_monotone_ratio': self.anti_smoothing_monotone_ratio,
                'anti_smoothing_oscillation_floor': self.anti_smoothing_oscillation_floor,
                'anti_smoothing_decay': self.anti_smoothing_decay,
                'anti_smoothing_target_amplitude_ratio': self.anti_smoothing_target_amplitude_ratio,
                'device': self.device,
            },
            series_length=len(train),
        )
        run_result = run_okhs_fdmd_forecaster_on_series(
            time_series=train,
            forecast_horizon=series_record.forecast_horizon,
            params=dict(spec.params),
        )
        forecast = np.asarray(
            run_result.forecast, dtype=float).reshape(-1)[:series_record.forecast_horizon]
        metadata = dict(run_result.diagnostics)
        metadata.update(
            {
                'selected_q': float(spec.params['q']),
                'window_size': int(spec.params['window_size']),
                'last_train_value': float(train[-1]),
                'first_prediction_value': float(forecast[0]) if len(forecast) else None,
                'first_actual_value': float(series_record.test_values[0]) if series_record.test_values else None,
            }
        )
        if metadata['first_prediction_value'] is not None:
            metadata['first_step_delta'] = float(
                metadata['first_prediction_value'] - metadata['last_train_value'])
        if metadata['first_actual_value'] is not None:
            metadata['first_actual_delta'] = float(
                metadata['first_actual_value'] - metadata['last_train_value'])
        metadata = _maybe_attach_stage_tuning_report(
            metadata,
            adapter_name='okhs_fdmd_forecaster',
            series_record=series_record,
            base_params=dict(spec.params),
            runtime_config=self.stage_tuning_runtime,
        )
        return forecast, metadata


@dataclass
class MSSAModel(ForecastingModelAdapter):
    """Benchmark adapter for the stage-aware MSSA forecaster."""

    window_size: int | None = None
    rank: int | None = None
    explained_variance: float = 0.95
    lag_order: int | None = None
    coupled: bool = False
    head_policy: str = 'mlp'
    head_hidden_dim: int = 64
    head_hidden_layers: int = 2
    head_epochs: int = 120
    head_learning_rate: float = 1e-3
    device: str = 'auto'
    progress_policy: dict[str, Any] | bool | None = None
    stage_tuning_runtime: dict[str, Any] | None = None
    name: str = 'mSSA'
    family: ModelFamily = ModelFamily.INTERNAL_INDUSTRIAL
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'mssa')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the local MSSA runtime."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit MSSA on train values and return forecast plus diagnostics."""
        train = np.asarray(series_record.train_values, dtype=float)
        model = MSSAForecaster(
            forecast_horizon=series_record.forecast_horizon,
            window_size=self.window_size,
            rank=self.rank,
            explained_variance=self.explained_variance,
            lag_order=self.lag_order,
            coupled=self.coupled,
            head_policy=self.head_policy,
            head_hidden_dim=self.head_hidden_dim,
            head_hidden_layers=self.head_hidden_layers,
            head_epochs=self.head_epochs,
            head_learning_rate=self.head_learning_rate,
            device=self.device,
            progress_policy=self.progress_policy,
        )
        model.fit(train)
        forecast = np.asarray(model.predict(train), dtype=float).reshape(-1)
        metadata = model.get_diagnostics()
        metadata = _maybe_attach_stage_tuning_report(
            metadata,
            adapter_name='mssa_forecaster',
            series_record=series_record,
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'lag_order': self.lag_order,
                'channel_independent': not self.coupled,
                'head_policy': self.head_policy,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'device': self.device,
                'progress_policy': self.progress_policy,
            },
            runtime_config={
                **dict(self.stage_tuning_runtime or {}),
                **(
                    {
                        'progress_policy': dict(self.progress_policy or {})
                        if isinstance(self.progress_policy, dict)
                        else self.progress_policy,
                    }
                    if self.progress_policy is not None else {}
                ),
            },
        )
        return forecast[:series_record.forecast_horizon], metadata


@dataclass
class SSACompatModel(ForecastingModelAdapter):
    """Benchmark adapter for the SSA compatibility wrapper."""

    window_size: int | None = None
    rank: int | None = None
    explained_variance: float = 0.95
    history_lookback: int = 0
    progress_policy: dict[str, Any] | bool | None = None
    stage_tuning_runtime: dict[str, Any] | None = None
    name: str = 'ssa_forecaster'
    family: ModelFamily = ModelFamily.INTERNAL_INDUSTRIAL
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'ssa')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether FEDOT compatibility runtime is available."""
        if not _safe_import('fedot.core.data.data'):
            return RunStatus.NOT_AVAILABLE, 'fedot is required for ssa_forecaster compatibility wrapper.'
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Run the SSA compatibility wrapper and return forecast metadata."""
        from fedot_ind.core.models.ts_forecasting.lagged_model.ssa_forecaster import SSAForecasterImplementation

        input_data = _build_fedot_forecasting_input(series_record)
        model = SSAForecasterImplementation(
            {
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'history_lookback': self.history_lookback,
                'mode': 'one_dimensional',
                'progress_policy': self.progress_policy,
            }
        )
        model.fit(input_data)
        output = model.predict(input_data)
        forecast = np.asarray(output.predict, dtype=float).reshape(-1)
        metadata: dict[str, Any] = {
            'compatibility_status': getattr(model, 'compatibility_status_', 'compatibility_wrapper'),
            'history_lookback': int(self.history_lookback),
            'mode': 'one_dimensional',
        }
        inner_model = getattr(model, 'model_', None)
        if inner_model is not None and hasattr(inner_model, 'get_diagnostics'):
            metadata.update(inner_model.get_diagnostics())
        metadata = _maybe_attach_stage_tuning_report(
            metadata,
            adapter_name='ssa_forecaster',
            series_record=series_record,
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'history_lookback': self.history_lookback,
                'mode': 'one_dimensional',
                'progress_policy': self.progress_policy,
            },
            runtime_config={
                **dict(self.stage_tuning_runtime or {}),
                **(
                    {
                        'progress_policy': dict(self.progress_policy or {})
                        if isinstance(self.progress_policy, dict)
                        else self.progress_policy,
                    }
                    if self.progress_policy is not None else {}
                ),
            },
        )
        return forecast[:series_record.forecast_horizon], metadata


@dataclass
class LaggedForecasterModel(ForecastingModelAdapter):
    """Benchmark adapter for lagged_forecaster via lagged ridge runtime."""

    channel_model: str = 'ridge'
    window_size: int = 10
    stride: int = 1
    alpha: float = 1.0
    stage_tuning_runtime: dict[str, Any] | None = None
    name: str = 'lagged_forecaster'
    family: ModelFamily = ModelFamily.INTERNAL_INDUSTRIAL
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'lagged_linear')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether lagged ridge runtime and channel model are supported."""
        if LaggedRidgeForecaster is None:
            return RunStatus.NOT_AVAILABLE, 'torch is required for lagged_ridge_forecaster runtime.'
        if str(self.channel_model).lower() != 'ridge':
            return RunStatus.NOT_AVAILABLE, (
                f"lagged_forecaster benchmark adapter currently supports only channel_model='ridge', "
                f"got '{self.channel_model}'."
            )
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit lagged ridge on train values and return forecast plus diagnostics."""
        if str(self.channel_model).lower() != 'ridge':
            raise ModelExecutionError(
                RunStatus.SKIPPED,
                f"lagged_forecaster benchmark adapter currently supports only channel_model='ridge', "
                f"got '{self.channel_model}'.",
            )
        model = LaggedRidgeForecaster(
            forecast_horizon=series_record.forecast_horizon,
            window_size_percent=self.window_size,
            stride=self.stride,
            alpha=self.alpha,
        )
        model.fit(np.asarray(series_record.train_values, dtype=float))
        forecast = np.asarray(model.predict(np.asarray(series_record.train_values, dtype=float)), dtype=float).reshape(
            -1)
        metadata = model.get_diagnostics()
        metadata.update(
            {
                'channel_model': str(self.channel_model),
                'window_size_percent': float(self.window_size),
                'stride': int(self.stride),
                'alpha': float(self.alpha),
            }
        )
        metadata = _maybe_attach_stage_tuning_report(
            metadata,
            adapter_name='lagged_forecaster',
            series_record=series_record,
            base_params={
                'window_size': self.window_size,
                'stride': self.stride,
                'channel_model': str(self.channel_model),
                'alpha': self.alpha,
            },
            runtime_config=self.stage_tuning_runtime,
        )
        return forecast[:series_record.forecast_horizon], metadata


@dataclass
class LowRankLaggedForecasterModel(ForecastingModelAdapter):
    """Benchmark adapter for low-rank lagged ridge forecasting."""

    window_size: int = 10
    stride: int = 1
    alpha: float = 1.0
    rank: int | None = None
    explained_variance: float = 0.95
    decomposition_strategy: str = 'full'
    rank_truncation_policy: str = 'explained_variance'
    stage_tuning_runtime: dict[str, Any] | None = None
    name: str = 'low_rank_lagged_ridge_forecaster'
    family: ModelFamily = ModelFamily.INTERNAL_INDUSTRIAL
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'low_rank_linear')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether the low-rank lagged runtime is available."""
        if LowRankLaggedRidgeForecaster is None:
            return RunStatus.NOT_AVAILABLE, 'torch is required for low_rank_lagged_ridge_forecaster runtime.'
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit low-rank lagged ridge and return forecast plus diagnostics."""
        model = LowRankLaggedRidgeForecaster(
            forecast_horizon=series_record.forecast_horizon,
            window_size_percent=self.window_size,
            stride=self.stride,
            alpha=self.alpha,
            rank=self.rank,
            explained_variance=self.explained_variance,
            decomposition_strategy=self.decomposition_strategy,
            rank_truncation_policy=self.rank_truncation_policy,
        )
        train = np.asarray(series_record.train_values, dtype=float)
        model.fit(train)
        forecast = np.asarray(model.predict(train), dtype=float).reshape(-1)
        metadata = model.get_diagnostics()
        metadata.update(
            {
                'window_size_percent': float(self.window_size),
                'stride': int(self.stride),
                'alpha': float(self.alpha),
            }
        )
        metadata = _maybe_attach_stage_tuning_report(
            metadata,
            adapter_name='low_rank_lagged_ridge_forecaster',
            series_record=series_record,
            base_params={
                'window_size': self.window_size,
                'stride': self.stride,
                'alpha': self.alpha,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'decomposition_strategy': self.decomposition_strategy,
                'rank_truncation_policy': self.rank_truncation_policy,
            },
            runtime_config=self.stage_tuning_runtime,
        )
        return forecast[:series_record.forecast_horizon], metadata


@dataclass
class HybridEnsembleModel(ForecastingModelAdapter):
    """Benchmark adapter for the named hybrid ensemble forecaster."""

    complex_branch: str = 'okhs'
    calibration_horizon: int | None = None
    lagged_params: dict[str, Any] = None
    low_rank_params: dict[str, Any] = None
    complex_params: dict[str, Any] = None
    stage_tuning_runtime: dict[str, Any] | None = None
    name: str = 'hybrid_ensemble_forecaster'
    family: ModelFamily = ModelFamily.INTERNAL_INDUSTRIAL
    tags: tuple[str, ...] = ('ensemble', 'forecasting', 'operator_model')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether the hybrid ensemble runtime is available."""
        if HybridEnsembleForecaster is None:
            return RunStatus.NOT_AVAILABLE, 'torch is required for hybrid_ensemble_forecaster runtime.'
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit all ensemble branches and return weighted forecast diagnostics."""
        model = HybridEnsembleForecaster(
            forecast_horizon=series_record.forecast_horizon,
            complex_branch=self.complex_branch,
            calibration_horizon=self.calibration_horizon,
            lagged_params=dict(self.lagged_params or {}),
            low_rank_params=dict(self.low_rank_params or {}),
            complex_params=dict(self.complex_params or {}),
        )
        train = np.asarray(series_record.train_values, dtype=float)
        model.fit(train)
        forecast = np.asarray(model.predict(train), dtype=float).reshape(-1)
        metadata = model.get_diagnostics()
        metadata.update({'complex_branch': self.complex_branch})
        metadata = _maybe_attach_stage_tuning_report(
            metadata,
            adapter_name='hybrid_ensemble_forecaster',
            series_record=series_record,
            base_params={
                'complex_branch': self.complex_branch,
                'calibration_horizon': self.calibration_horizon,
                'lagged_params': dict(self.lagged_params or {}),
                'low_rank_params': dict(self.low_rank_params or {}),
                'complex_params': dict(self.complex_params or {}),
            },
            runtime_config=self.stage_tuning_runtime,
        )
        return forecast[:series_record.forecast_horizon], metadata


@dataclass
class HAVOKModel(ForecastingModelAdapter):
    """Benchmark adapter for the stage-aware HAVOK forecaster."""

    window_size: int | None = None
    rank: int | None = None
    forcing_threshold_scale: float = 1.0
    forcing_decay: float = 0.85
    head_policy: str = 'mlp'
    head_activation: str = 'relu'
    head_depth: int = 2
    head_base_hidden_dim: int = 512
    head_hidden_dim: int | None = None
    head_hidden_layers: int | None = None
    head_epochs: int = 120
    head_learning_rate: float = 1e-3
    device: str = 'auto'
    progress_policy: dict[str, Any] | bool | None = None
    stage_tuning_runtime: dict[str, Any] | None = None
    name: str = 'HAVOK'
    family: ModelFamily = ModelFamily.INTERNAL_INDUSTRIAL
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'havok')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the local HAVOK runtime."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit HAVOK on train values and return forecast plus diagnostics."""
        train = np.asarray(series_record.train_values, dtype=float)
        model = HAVOKForecaster(
            forecast_horizon=series_record.forecast_horizon,
            window_size=self.window_size,
            rank=self.rank,
            forcing_threshold_scale=self.forcing_threshold_scale,
            forcing_decay=self.forcing_decay,
            head_policy=self.head_policy,
            head_activation=self.head_activation,
            head_depth=self.head_depth,
            head_base_hidden_dim=self.head_base_hidden_dim,
            head_hidden_dim=self.head_hidden_dim,
            head_hidden_layers=self.head_hidden_layers,
            head_epochs=self.head_epochs,
            head_learning_rate=self.head_learning_rate,
            device=self.device,
            progress_policy=self.progress_policy,
        )
        model.fit(train)
        forecast = np.asarray(model.predict(train), dtype=float).reshape(-1)
        metadata = model.get_diagnostics()
        metadata.update(
            {
                'last_train_value': float(train[-1]),
                'first_prediction_value': float(forecast[0]) if len(forecast) else None,
                'first_actual_value': float(series_record.test_values[0]) if series_record.test_values else None,
            }
        )
        metadata = _maybe_attach_stage_tuning_report(
            metadata,
            adapter_name='havok_forecaster',
            series_record=series_record,
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'forcing_threshold_scale': self.forcing_threshold_scale,
                'forcing_decay': self.forcing_decay,
                'head_policy': self.head_policy,
                'head_activation': self.head_activation,
                'head_depth': self.head_depth,
                'head_base_hidden_dim': self.head_base_hidden_dim,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'device': self.device,
                'progress_policy': self.progress_policy,
            },
            runtime_config={
                **dict(self.stage_tuning_runtime or {}),
                **(
                    {
                        'progress_policy': dict(self.progress_policy or {})
                        if isinstance(self.progress_policy, dict)
                        else self.progress_policy,
                    }
                    if self.progress_policy is not None else {}
                ),
            },
        )
        return forecast[:series_record.forecast_horizon], metadata


@dataclass
class NeuralForecastingHeadModel(ForecastingModelAdapter):
    """Benchmark adapter for primitive neural forecasting heads."""

    neural_model_name: str = 'patch_tst_model'
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    activation: str | None = None
    patch_len: int | None = None
    forecast_mode: str | None = None
    use_amp: bool | None = None
    model_dim: int | None = None
    n_layers: int | None = None
    number_heads: int | None = None
    d_ff: int | None = None
    kernel_size: int | None = None
    num_filters: int | None = None
    num_layers: int | None = None
    dilation_base: int | None = None
    dropout: float | None = None
    weight_norm: bool | None = None
    cell_type: str | None = None
    rnn_layers: int | None = None
    hidden_size: int | None = None
    expected_distribution: str | None = None
    n_stacks: int | None = None
    n_trend_blocks: int | None = None
    n_seasonality_blocks: int | None = None
    n_of_harmonics: int | None = None
    layers: int | None = None
    degree_of_polynomial: int | None = None
    stage_tuning_runtime: dict[str, Any] | None = None
    name: str = 'NeuralForecastHead'
    family: ModelFamily = ModelFamily.SUPERVISED_SOTA
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'neural_forecaster')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether torch and neural head runtime are available."""
        if torch is None or run_neural_forecast_head_on_series is None:
            return RunStatus.NOT_AVAILABLE, 'torch/native neural forecasting runtime is unavailable.'
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Run a neural forecasting head on train values and return diagnostics."""
        train = np.asarray(series_record.train_values, dtype=float)
        params = {
            key: value for key, value in self.__dict__.items()
            if key not in {'name', 'tags', 'optional', 'stage_tuning_runtime', 'neural_model_name'}
        }
        run_result = run_neural_forecast_head_on_series(
            self.neural_model_name,
            time_series=train,
            forecast_horizon=series_record.forecast_horizon,
            params=params,
        )
        forecast = np.asarray(run_result.forecast, dtype=float).reshape(-1)
        metadata = dict(run_result.diagnostics)
        metadata.update(
            {
                'last_train_value': float(train[-1]),
                'first_prediction_value': float(forecast[0]) if len(forecast) else None,
                'first_actual_value': float(series_record.test_values[0]) if series_record.test_values else None,
            }
        )
        metadata = _maybe_attach_stage_tuning_report(
            metadata,
            adapter_name=self.neural_model_name,
            series_record=series_record,
            base_params=params,
            runtime_config=self.stage_tuning_runtime,
        )
        return forecast[:series_record.forecast_horizon], metadata


@dataclass
class OptionalExternalModel(ForecastingModelAdapter):
    """Placeholder adapter for optional external forecasting backends."""

    dependency_name: str
    name: str
    family: ModelFamily = ModelFamily.EXTERNAL
    tags: tuple[str, ...] = ('baseline', 'forecasting', 'external')
    optional: bool = True
    scaffold_reason: str = 'Adapter scaffold is registered but backend training is not wired yet.'

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether the optional external dependency can be imported."""
        if not _safe_import(self.dependency_name):
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'
        return RunStatus.SUCCESS, 'dependency is available'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Raise a skipped execution status until the external backend is wired."""
        del series_record
        raise ModelExecutionError(RunStatus.SKIPPED, self.scaffold_reason)


def build_dataset_adapter(spec: DatasetSpec):
    """Create a dataset adapter for a forecasting dataset specification."""
    benchmark = spec.benchmark.lower()
    custom_loader = spec.adapter_options.get('loader')
    if benchmark == 'm4':
        return M4Adapter(loader=custom_loader)
    if benchmark == 'monash':
        return MonashAdapter(loader=custom_loader)
    if benchmark == 'in_memory':
        return InMemoryForecastingAdapter()
    raise BenchmarkConfigurationError(
        f'Unsupported forecasting benchmark adapter: {spec.benchmark}')


def _instantiate_model_adapter(adapter_cls, spec: ModelSpec, default_tags: tuple[str, ...]):
    accepted_parameters = set(inspect.signature(adapter_cls).parameters)
    filtered_params = {
        key: value
        for key, value in dict(spec.params).items()
        if key in accepted_parameters
    }
    return adapter_cls(
        name=spec.display_name,
        tags=spec.tags or default_tags,
        **filtered_params,
    )


def build_model_adapter(spec: ModelSpec) -> ForecastingModelAdapter:
    """Create a forecasting model adapter from a benchmark model spec."""
    raw_adapter_name = spec.adapter_name.lower()
    adapter_name = canonical_forecasting_model_name(raw_adapter_name)
    dict(spec.params)
    if raw_adapter_name == 'okhs':
        return _instantiate_model_adapter(OKHSModel, spec, ('okhs', 'forecasting'))
    if adapter_name == 'okhs_fdmd_forecaster':
        return _instantiate_model_adapter(OKHSFDMDForecasterModel, spec, ('okhs', 'forecasting', 'operator_model'))
    if adapter_name == 'ssa_forecaster':
        return _instantiate_model_adapter(SSACompatModel, spec, ('baseline', 'forecasting', 'ssa'))
    if adapter_name == 'lagged_forecaster':
        return _instantiate_model_adapter(LaggedForecasterModel, spec, ('baseline', 'forecasting', 'lagged_linear'))
    if adapter_name == 'lagged_ridge_forecaster':
        return _instantiate_model_adapter(LaggedForecasterModel, spec, ('baseline', 'forecasting', 'lagged_linear'))
    if adapter_name == 'low_rank_lagged_ridge_forecaster':
        return _instantiate_model_adapter(LowRankLaggedForecasterModel, spec,
                                          ('baseline', 'forecasting', 'low_rank_linear'))
    if adapter_name == 'hybrid_ensemble_forecaster':
        return _instantiate_model_adapter(HybridEnsembleModel, spec, ('ensemble', 'forecasting', 'operator_model'))
    if adapter_name in {'mssa', 'mssa_forecaster'}:
        return _instantiate_model_adapter(MSSAModel, spec, ('baseline', 'forecasting', 'mssa'))
    if adapter_name in {'havok', 'havok_forecaster'}:
        return _instantiate_model_adapter(HAVOKModel, spec, ('baseline', 'forecasting', 'havok'))
    if adapter_name in {'patch_tst_model', 'tst_model', 'tcn_model', 'deepar_model', 'nbeats_model'}:
        filtered_params = {
            key: value
            for key, value in dict(spec.params).items()
            if key in set(inspect.signature(NeuralForecastingHeadModel).parameters)
        }
        return NeuralForecastingHeadModel(
            neural_model_name=adapter_name,
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'forecasting', 'neural_forecaster'),
            **filtered_params,
        )
    if adapter_name == 'naive_last_value':
        return NaiveLastValueModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'))
    if adapter_name == 'naive_mean':
        return NaiveMeanModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'))
    if adapter_name == 'naive_drift':
        return NaiveDriftModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'))
    if adapter_name == 'moving_average':
        return _instantiate_model_adapter(MovingAverageModel, spec, ('baseline', 'forecasting'))
    if adapter_name == 'linear_trend':
        return LinearTrendModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting'))
    if adapter_name == 'classical_dmd':
        return _instantiate_model_adapter(ClassicalDMDModel, spec, ('baseline', 'forecasting', 'dmd'))
    if adapter_name == 'autogluon':
        return OptionalExternalModel(
            dependency_name='autogluon',
            name=spec.display_name,
            family=ModelFamily.AUTOML,
            tags=spec.tags or ('baseline', 'forecasting',
                               'external', 'autogluon'),
        )
    if adapter_name == 'nbeats':
        return OptionalExternalModel(
            dependency_name='neuralforecast',
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'forecasting',
                               'external', 'nbeats'),
        )
    if adapter_name == 'tft':
        return OptionalExternalModel(
            dependency_name='pytorch_forecasting',
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'forecasting', 'external', 'tft'),
        )

    if spec.family is not None:
        adapter.family = spec.family

    raise BenchmarkConfigurationError(
        f'Unsupported forecasting model adapter: {spec.adapter_name}')


def _seasonal_naive_forecast(train: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    lag = seasonal_period if seasonal_period > 1 and len(
        train) > seasonal_period else 1
    base = train[-lag:]
    repeats = int(math.ceil(horizon / lag))
    return np.tile(base, repeats)[:horizon]


def _mase_scale(train: np.ndarray, seasonal_period: int) -> float:
    lag = seasonal_period if seasonal_period > 1 and len(
        train) > seasonal_period else 1
    if len(train) <= lag:
        return 1.0
    scale = np.mean(np.abs(train[lag:] - train[:-lag]))
    return float(scale if scale > 1e-8 else 1.0)


def compute_forecasting_metric(
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int,
) -> float:
    """Compute an aggregate forecasting metric for one horizon vector."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    train = np.asarray(y_train, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise BenchmarkConfigurationError(
            'Metric inputs must have the same length.')
    if metric_name == 'mae':
        return float(np.mean(np.abs(actual - predicted)))
    if metric_name == 'rmse':
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    if metric_name == 'smape':
        denominator = np.abs(actual) + np.abs(predicted)
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return float(100.0 * np.mean(2.0 * np.abs(predicted - actual) / denominator))
    if metric_name == 'mase':
        return float(np.mean(np.abs(actual - predicted)) / _mase_scale(train, seasonal_period))
    if metric_name == 'owa':
        baseline = _seasonal_naive_forecast(
            train, len(actual), seasonal_period)
        smape = compute_forecasting_metric(
            'smape', actual, predicted, train, seasonal_period)
        mase = compute_forecasting_metric(
            'mase', actual, predicted, train, seasonal_period)
        baseline_smape = compute_forecasting_metric(
            'smape', actual, baseline, train, seasonal_period)
        baseline_mase = compute_forecasting_metric(
            'mase', actual, baseline, train, seasonal_period)
        baseline_smape = baseline_smape if baseline_smape > 1e-8 else 1.0
        baseline_mase = baseline_mase if baseline_mase > 1e-8 else 1.0
        return float(0.5 * ((smape / baseline_smape) + (mase / baseline_mase)))
    raise BenchmarkConfigurationError(f'Unsupported metric: {metric_name}')


def compute_pointwise_metric(
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int,
) -> np.ndarray:
    """Compute horizon-wise metric values for publication artifacts."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if metric_name == 'mae':
        return np.abs(actual - predicted)
    if metric_name == 'rmse':
        return np.sqrt((actual - predicted) ** 2)
    if metric_name == 'smape':
        denominator = np.abs(actual) + np.abs(predicted)
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return 100.0 * 2.0 * np.abs(predicted - actual) / denominator
    if metric_name == 'mase':
        return np.abs(actual - predicted) / _mase_scale(np.asarray(y_train, dtype=float), seasonal_period)
    if metric_name == 'owa':
        train = np.asarray(y_train, dtype=float)
        baseline = _seasonal_naive_forecast(
            train, len(actual), seasonal_period)
        pointwise_smape = compute_pointwise_metric(
            'smape', actual, predicted, train, seasonal_period)
        pointwise_mase = compute_pointwise_metric(
            'mase', actual, predicted, train, seasonal_period)
        baseline_smape = compute_pointwise_metric(
            'smape', actual, baseline, train, seasonal_period)
        baseline_mase = compute_pointwise_metric(
            'mase', actual, baseline, train, seasonal_period)
        baseline_smape = np.where(baseline_smape <= 1e-8, 1.0, baseline_smape)
        baseline_mase = np.where(baseline_mase <= 1e-8, 1.0, baseline_mase)
        return 0.5 * ((pointwise_smape / baseline_smape) + (pointwise_mase / baseline_mase))
    raise BenchmarkConfigurationError(f'Unsupported metric: {metric_name}')


def _extract_forecast_event_mask(metadata: dict[str, Any], horizon: int) -> np.ndarray | None:
    raw_mask = metadata.get('forecast_forcing_mask')
    if raw_mask is None:
        return None
    mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return None
    if mask.size < horizon:
        mask = np.pad(mask, (0, horizon - mask.size), constant_values=False)
    return mask[:horizon]


def _append_event_interval_metrics(
        metric_records: list[MetricRecord],
        *,
        run_id: str,
        series_record: ForecastingSeriesRecord,
        model_name: str,
        actual: np.ndarray,
        forecast: np.ndarray,
        metadata: dict[str, Any],
        metric_name_suffix: str = '',
) -> dict[str, float]:
    mask = _extract_forecast_event_mask(metadata, len(actual))
    if mask is None:
        return {}
    pointwise_mae = np.abs(np.asarray(
        actual, dtype=float).reshape(-1) - np.asarray(forecast, dtype=float).reshape(-1))
    event_metrics: dict[str, float] = {}
    for suffix, subset_mask in (('active', mask), ('calm', ~mask)):
        if not np.any(subset_mask):
            continue
        metric_name = f'mae_{suffix}'
        metric_value = float(np.mean(pointwise_mae[subset_mask]))
        event_metrics[metric_name] = metric_value
        metric_records.append(
            MetricRecord(
                run_id=run_id,
                benchmark=series_record.benchmark,
                dataset_name=series_record.dataset_name,
                subset=series_record.subset,
                series_id=series_record.series_id,
                model_name=model_name,
                metric_name=f'{metric_name}{metric_name_suffix}',
                metric_value=metric_value,
                status=RunStatus.SUCCESS,
            )
        )
    event_metrics['active_forecast_steps'] = float(int(np.sum(mask)))
    event_metrics['calm_forecast_steps'] = float(int(np.sum(~mask)))
    return event_metrics


@dataclass
class ForecastingSeriesArtifactsRecorder:
    """Collect per-series metric and prediction artifacts during a run."""

    run_id: str
    metric_names: tuple[str, ...]
    metric_records: list[MetricRecord]
    prediction_records: list[PredictionRecord]
    quantile_prediction_records: list[QuantilePredictionRecord]

    def validate_forecast_length(
            self,
            series_record: ForecastingSeriesRecord,
            prediction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate model forecast length and return aligned actual/forecast arrays."""
        actual = np.asarray(series_record.test_values, dtype=float)
        forecast = np.asarray(
            prediction, dtype=float).reshape(-1)[: len(actual)]
        if len(forecast) != len(actual):
            raise ModelExecutionError(
                RunStatus.FAILED,
                f'Model returned {len(forecast)} predictions for horizon {len(actual)}.',
            )
        return actual, forecast

    def record_metric_bundle(
            self,
            *,
            series_record: ForecastingSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            forecast: np.ndarray,
            metadata: dict[str, Any],
            metric_name_suffix: str = '',
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Record aggregate, horizon-wise and event-aware metric rows."""
        metrics_summary: dict[str, float] = {}
        train = np.asarray(series_record.train_values, dtype=float)
        for metric_name in self.metric_names:
            metric_value = compute_forecasting_metric(
                metric_name,
                actual,
                forecast,
                train,
                series_record.seasonal_period,
            )
            metrics_summary[metric_name] = metric_value
            self.metric_records.append(
                MetricRecord(
                    run_id=self.run_id,
                    benchmark=series_record.benchmark,
                    dataset_name=series_record.dataset_name,
                    subset=series_record.subset,
                    series_id=series_record.series_id,
                    model_name=model_name,
                    metric_name=f'{metric_name}{metric_name_suffix}',
                    metric_value=metric_value,
                    status=RunStatus.SUCCESS,
                )
            )
            pointwise = compute_pointwise_metric(
                metric_name,
                actual,
                forecast,
                train,
                series_record.seasonal_period,
            )
            for horizon_index, pointwise_value in enumerate(pointwise, start=1):
                self.metric_records.append(
                    MetricRecord(
                        run_id=self.run_id,
                        benchmark=series_record.benchmark,
                        dataset_name=series_record.dataset_name,
                        subset=series_record.subset,
                        series_id=series_record.series_id,
                        model_name=model_name,
                        metric_name=f'{metric_name}{metric_name_suffix}',
                        metric_value=float(pointwise_value),
                        status=RunStatus.SUCCESS,
                        horizon_index=horizon_index,
                    )
                )

        event_metrics = _append_event_interval_metrics(
            self.metric_records,
            run_id=self.run_id,
            series_record=series_record,
            model_name=model_name,
            actual=actual,
            forecast=forecast,
            metadata=metadata,
            metric_name_suffix=metric_name_suffix,
        )
        metrics_summary.update(
            {key: value for key, value in event_metrics.items() if key.startswith('mae_')})
        return metrics_summary, event_metrics

    def record_predictions(
            self,
            *,
            series_record: ForecastingSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            forecast: np.ndarray,
    ) -> None:
        """Append per-horizon prediction records for one series/model item."""
        for horizon_index, (actual_value, forecast_value) in enumerate(zip(actual, forecast), start=1):
            self.prediction_records.append(
                PredictionRecord(
                    run_id=self.run_id,
                    benchmark=series_record.benchmark,
                    dataset_name=series_record.dataset_name,
                    subset=series_record.subset,
                    series_id=series_record.series_id,
                    model_name=model_name,
                    horizon_index=horizon_index,
                    y_true=float(actual_value),
                    y_pred=float(forecast_value),
                    status=RunStatus.SUCCESS,
                )
            )

    def record_quantile_predictions(
            self,
            series_record: ForecastingSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            forecast_result: ForecastResult,
    ) -> None:
        self.quantile_prediction_records.extend(
            iter_quantile_prediction_records(
                run_id=self.run_id,
                series_record=series_record,
                model_name=model_name,
                actual=actual,
                forecast_result=forecast_result,
            )
        )


@dataclass
class ForecastingPostFitTuningCoordinator:
    """Run mandatory post-fit stage tuning and compare tuned vs baseline metrics."""

    config: BenchmarkSuiteConfig
    verbosity_policy: ForecastingVerbosityPolicy
    artifacts_recorder: ForecastingSeriesArtifactsRecorder

    def _build_tuned_model_spec(self, model_spec: ModelSpec, best_parameters: dict[str, Any]) -> ModelSpec:
        tuned_params = {
            **dict(model_spec.params),
            **dict(best_parameters),
        }
        tuned_params.pop('stage_tuning_runtime', None)
        return ModelSpec(
            adapter_name=model_spec.adapter_name,
            display_name=model_spec.display_name,
            tags=model_spec.tags,
            optional=model_spec.optional,
            params=tuned_params,
        )

    def _resolve_runtime_config(
            self,
            model_spec: ModelSpec,
            baseline_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        raw_config = dict(model_spec.params.get('stage_tuning_runtime') or {})
        metadata_runtime = dict(
            baseline_metadata.get('stage_tuning_runtime') or {})
        raw_progress_policy = raw_config.get(
            'progress_policy', metadata_runtime.get('progress_policy'))
        raw_verbosity_policy = raw_config.get(
            'verbosity_policy', self.verbosity_policy.to_dict())
        if raw_progress_policy is None:
            resolved_progress_policy = resolve_forecasting_progress_policy(
                None,
                show_progress=self.config.run_spec.show_progress,
            )
        else:
            resolved_progress_policy = resolve_forecasting_progress_policy(
                raw_progress_policy)
        resolved_verbosity_policy = (
            raw_verbosity_policy
            if isinstance(raw_verbosity_policy, ForecastingVerbosityPolicy)
            else resolve_forecasting_verbosity_policy(
                (raw_verbosity_policy or {}).get('level') if isinstance(raw_verbosity_policy, dict)
                else raw_verbosity_policy,
                options=raw_verbosity_policy if isinstance(
                    raw_verbosity_policy, dict) else None,
            )
        )
        return {
            'metric_name': str(
                raw_config.get(
                    'metric_name',
                    metadata_runtime.get(
                        'metric_name',
                        self.config.run_spec.primary_metric))),
            'stage_updates': raw_config.get('stage_updates'),
            'max_values_per_parameter': int(
                raw_config.get(
                    'max_values_per_parameter',
                    3)),
            'max_stage_candidates': int(
                raw_config.get(
                    'max_stage_candidates',
                    16)),
            'split_spec': _resolve_stage_tuning_split_spec(raw_config),
            'progress_policy': resolved_progress_policy,
            'verbosity_policy': resolved_verbosity_policy,
        }

    def run(
            self,
            *,
            model_spec: ModelSpec,
            model: ForecastingModelAdapter,
            series_record: ForecastingSeriesRecord,
            baseline_metadata: dict[str, Any],
            baseline_metrics_summary: dict[str, float],
            regime_diagnostics,
            routing_recommendation,
    ) -> dict[str, Any]:
        """Execute stage tuning after baseline validation and return enriched metadata."""
        runtime_config = self._resolve_runtime_config(
            model_spec, baseline_metadata)
        if run_forecasting_stage_tuning_on_series is None:
            return {
                **baseline_metadata,
                'stage_tuning_report_error': 'stage_tuning_runtime is unavailable in the current environment.',
            }

        try:
            report = run_forecasting_stage_tuning_on_series(
                model_spec.adapter_name,
                time_series=np.asarray(
                    series_record.train_values, dtype=float),
                forecast_horizon=series_record.forecast_horizon,
                base_params={
                    key: value
                    for key, value in dict(model_spec.params).items()
                    if key != 'stage_tuning_runtime'
                },
                stage_updates=runtime_config.get('stage_updates'),
                metric_name=str(runtime_config['metric_name']),
                split_spec=runtime_config.get('split_spec'),
                seasonal_period=series_record.seasonal_period,
                max_values_per_parameter=int(
                    runtime_config['max_values_per_parameter']),
                max_stage_candidates=int(
                    runtime_config['max_stage_candidates']),
                progress_policy=runtime_config.get('progress_policy'),
            )
            verbosity_policy = runtime_config.get(
                'verbosity_policy', self.verbosity_policy)
            report_dict = verbosity_policy.prune_stage_tuning_report(
                report.to_dict()) or {}
            sequential_result = dict(
                report_dict.get('sequential_result') or {})
            best_parameters = dict(
                sequential_result.get('best_parameters') or {})
            enriched_baseline_metadata = {
                **baseline_metadata,
                'stage_tuning_report': report_dict,
                'stage_tuning_runtime': verbosity_policy.prune_stage_tuning_runtime({
                    'enabled': True,
                    'metric_name': str(runtime_config['metric_name']),
                    'max_values_per_parameter': int(runtime_config['max_values_per_parameter']),
                    'max_stage_candidates': int(runtime_config['max_stage_candidates']),
                    'improved': report.metadata.get('improved'),
                    'baseline_score': report.metadata.get('baseline_score'),
                    'best_score': report.metadata.get('best_score'),
                    'progress_policy': resolve_forecasting_progress_policy(
                        runtime_config.get('progress_policy'),
                        show_progress=self.config.run_spec.show_progress,
                    ).to_dict(),
                }) or {},
            }
            if not best_parameters:
                return enriched_baseline_metadata

            tuned_model_spec = self._build_tuned_model_spec(
                model_spec, best_parameters)
            tuned_model = build_model_adapter(tuned_model_spec)
            tuned_status, tuned_message = tuned_model.availability()
            if tuned_status is not RunStatus.SUCCESS:
                return {
                    **enriched_baseline_metadata,
                    'stage_tuning_comparison_error': tuned_message,
                }

            tuned_prediction, tuned_metadata = tuned_model.forecast(
                series_record)
            actual, tuned_forecast = self.artifacts_recorder.validate_forecast_length(
                series_record, tuned_prediction)
            tuned_metrics_summary, tuned_event_metrics = self.artifacts_recorder.record_metric_bundle(
                series_record=series_record,
                model_name=model.name,
                actual=actual,
                forecast=tuned_forecast,
                metadata=tuned_metadata,
                metric_name_suffix='_tuned',
            )
            tuned_metrics_summary.update(
                {key: value for key, value in tuned_event_metrics.items()
                 if key.startswith('mae_')}
            )
            comparison_payload = verbosity_policy.prune_stage_tuning_comparison(
                {
                    'best_parameters': best_parameters,
                    'baseline_metrics': baseline_metrics_summary,
                    'tuned_metrics': tuned_metrics_summary,
                    'improved_metrics': {
                        metric_name: bool(
                            tuned_metrics_summary.get(metric_name, math.inf)
                            <= baseline_metrics_summary.get(metric_name, math.inf)
                        )
                        for metric_name in baseline_metrics_summary
                        if metric_name in tuned_metrics_summary
                    },
                    'absolute_gain': {
                        metric_name: float(
                            baseline_metrics_summary[metric_name] - tuned_metrics_summary[metric_name])
                        for metric_name in baseline_metrics_summary
                        if metric_name in tuned_metrics_summary
                    },
                    'tuned_metadata': dict(tuned_metadata),
                    'tuned_forecast': [float(value) for value in tuned_forecast.tolist()],
                    'tuned_adapter_family': adapter_name_to_family(tuned_model_spec.adapter_name),
                    'regime_diagnostics': regime_diagnostics.to_dict(),
                    'routing_recommendation': routing_recommendation.to_dict(),
                }
            )
            return {
                **enriched_baseline_metadata,
                **({'stage_tuning_comparison': comparison_payload} if comparison_payload is not None else {}),
            }
        except Exception as exc:
            return {
                **baseline_metadata,
                'stage_tuning_comparison_error': str(exc),
            }


def build_leaderboard(
        run_records: tuple[BenchmarkRunRecord, ...],
        primary_metric: str,
) -> BenchmarkAggregateReport:
    """Aggregate successful run records into a benchmark leaderboard."""
    successful = [
        record for record in run_records if record.status is RunStatus.SUCCESS]
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for record in successful:
        metric_value = record.metrics_summary.get(primary_metric)
        family = record.family if hasattr(record, 'family') else 'unknown'
        if metric_value is None:
            continue
        grouped.setdefault((record.benchmark, record.dataset_name,
                           record.model_name), []).append(metric_value)

    leaderboard_rows = []
    for (benchmark, dataset_name, model_name), values in grouped.items():
        leaderboard_rows.append(
            {
                'benchmark': benchmark,
                'dataset_name': dataset_name,
                'model_name': model_name,
                'family': family,
                primary_metric: float(np.mean(values)),
                'n_series': len(values),
            }
        )

    leaderboard_rows = sorted(
        leaderboard_rows, key=lambda row: row[primary_metric])
    for rank, row in enumerate(leaderboard_rows, start=1):
        row['rank'] = rank

    status_counts: dict[str, int] = {}
    for record in run_records:
        status_counts[record.status.value] = status_counts.get(
            record.status.value, 0) + 1

    run_id = run_records[0].run_id if run_records else new_run_id('empty')
    return BenchmarkAggregateReport(
        run_id=run_id,
        task_type=TaskType.FORECASTING,
        primary_metric=primary_metric,
        leaderboard_rows=tuple(leaderboard_rows),
        status_counts=status_counts,
    )


class ForecastingSuiteRunner:
    """Orchestrate forecasting benchmark datasets, models, series and artifacts."""

    def __init__(self, config: BenchmarkSuiteConfig):
        """Initialize benchmark state, progress, verbosity and resume coordinators."""
        validate_forecasting_suite_config(config)
        self.config = config
        self.run_id = (
            ForecastingIncrementalPersistenceCoordinator.resolve_run_id(config)
            or new_run_id(config.run_spec.run_name)
        )
        self.series_records: list[ForecastingSeriesRecord] = []
        self.run_records: list[BenchmarkRunRecord] = []
        self.prediction_records: list[PredictionRecord] = []
        self.metric_records: list[MetricRecord] = []
        self.quantile_prediction_records: list[QuantilePredictionRecord] = []
        self.known_series_keys: set[tuple[str, str, str, str]] = set()
        self.resumed_item_keys: set[str] = set()
        self.progress_policy = resolve_forecasting_progress_policy(
            ForecastingProgressPolicy(
                enabled=bool(config.run_spec.show_progress),
                leave=bool(config.run_spec.progress_leave),
                stage_tuning_enabled=bool(config.run_spec.show_progress),
                head_training_enabled=bool(config.run_spec.show_progress),
            ),
            show_progress=config.run_spec.show_progress,
        )
        self.verbosity_policy = resolve_forecasting_verbosity_policy(
            config.run_spec.verbosity,
            options=config.run_spec.verbosity_options,
        )
        self.artifacts_recorder = ForecastingSeriesArtifactsRecorder(
            run_id=self.run_id,
            metric_names=tuple(self.config.metrics),
            metric_records=self.metric_records,
            prediction_records=self.prediction_records,
            quantile_prediction_records=self.quantile_prediction_records,
        )
        self.post_fit_tuning = ForecastingPostFitTuningCoordinator(
            config=self.config,
            verbosity_policy=self.verbosity_policy,
            artifacts_recorder=self.artifacts_recorder,
        )
        self.progress = BenchmarkProgressMonitor(
            enabled=config.run_spec.show_progress,
            task_type=config.task_type.value,
            run_name=config.run_spec.run_name,
            leave=config.run_spec.progress_leave,
            log_errors=config.run_spec.progress_log_errors,
            log_summaries=config.run_spec.progress_log_summaries,
        )
        self.incremental_persistence = ForecastingIncrementalPersistenceCoordinator(
            config=self.config,
            run_id=self.run_id,
        )
        self._load_resume_state()

    def _series_key(self, series_record: ForecastingSeriesRecord) -> tuple[str, str, str, str]:
        return (
            str(series_record.benchmark),
            str(series_record.dataset_name),
            str(series_record.subset),
            str(series_record.series_id),
        )

    def _load_resume_state(self) -> None:
        resume_state = self.incremental_persistence.load_resume_state()
        if resume_state is None:
            return
        self.series_records = list(resume_state.series_records)
        self.run_records = list(resume_state.run_records)
        self.metric_records = list(resume_state.metric_records)
        self.prediction_records = list(resume_state.prediction_records)
        self.quantile_prediction_records = list(
            resume_state.quantile_prediction_records)
        self.known_series_keys = {self._series_key(
            record) for record in self.series_records}
        self.resumed_item_keys = set(resume_state.item_artifact_paths)
        self.progress.seed_resume_state(
            completed_items=resume_state.completed_items,
            status_counts=resume_state.status_counts,
        )
        self._sync_artifacts_recorder_lists()

    def _sync_artifacts_recorder_lists(self) -> None:
        self.artifacts_recorder.metric_records = self.metric_records
        self.artifacts_recorder.prediction_records = self.prediction_records
        self.artifacts_recorder.quantile_prediction_records = self.quantile_prediction_records

    def _augment_model_spec_with_progress_policy(self, model_spec: ModelSpec) -> ModelSpec:
        params = dict(model_spec.params)
        params.setdefault('progress_policy', self.progress_policy.to_dict())
        if isinstance(params.get('stage_tuning_runtime'), dict):
            params['stage_tuning_runtime'] = {
                **dict(params['stage_tuning_runtime']),
                'verbosity_policy': dict(
                    params['stage_tuning_runtime'].get(
                        'verbosity_policy', self.verbosity_policy.to_dict())
                ),
            }
        return ModelSpec(
            adapter_name=model_spec.adapter_name,
            display_name=model_spec.display_name,
            tags=model_spec.tags,
            optional=model_spec.optional,
            params=params,
        )

    def _runner_context_metadata(self) -> dict[str, Any]:
        if not self.verbosity_policy.include_runner_context:
            return {}
        return {
            'benchmark_runtime_context': {
                'progress_policy': self.progress_policy.to_dict(),
                'verbosity_policy': self.verbosity_policy.to_dict(),
            }
        }

    def run_suite(self) -> ForecastingBenchmarkResult:
        """Run the configured forecasting suite and return all collected records."""
        try:
            self._iter_over_datasets()
        finally:
            self.progress.close()

        aggregate_report = build_leaderboard(
            tuple(self.run_records),
            primary_metric=self.config.run_spec.primary_metric,
        )
        return ForecastingBenchmarkResult(
            run_id=self.run_id,
            config=self.config,
            series_records=tuple(self.series_records),
            run_records=tuple(self.run_records),
            prediction_records=tuple(self.prediction_records),
            quantile_prediction_records=tuple(
                self.quantile_prediction_records),
            metric_records=tuple(self.metric_records),
            aggregate_report=aggregate_report,
            artifact_manifest=self.incremental_persistence.build_artifact_manifest(),
            scenario_spec=self.config.scenario_spec,
        )

    def _iter_over_datasets(self) -> None:
        for dataset_spec in self.config.datasets:
            dataset_series = self._load_dataset_series(dataset_spec)
            self._iter_over_models(dataset_spec, dataset_series)
            self.progress.dataset_finished()

    def _load_dataset_series(self, dataset_spec: DatasetSpec) -> tuple[ForecastingSeriesRecord, ...]:
        dataset_adapter = build_dataset_adapter(dataset_spec)
        dataset_series = dataset_adapter.load_series(dataset_spec)
        for record in dataset_series:
            record_key = self._series_key(record)
            if record_key not in self.known_series_keys:
                self.series_records.append(record)
                self.known_series_keys.add(record_key)
        self.incremental_persistence.persist_series_catalog(
            self.series_records)
        self.progress.extend_total(
            len(dataset_series) * len(self.config.models))
        self.progress.dataset_loaded(
            dataset_spec.dataset_name, len(dataset_series))
        return dataset_series

    def _iter_over_models(
            self,
            dataset_spec: DatasetSpec,
            dataset_series: tuple[ForecastingSeriesRecord, ...],
    ) -> None:
        for model_spec in self.config.models:
            resolved_model_spec = self._augment_model_spec_with_progress_policy(
                model_spec)
            model = build_model_adapter(resolved_model_spec)
            self.progress.model_started(dataset_spec.dataset_name, model.name)
            try:
                availability_status, availability_message = model.availability()
                if availability_status is not RunStatus.SUCCESS:
                    self._handle_unavailable_model(
                        model_spec=resolved_model_spec,
                        model=model,
                        dataset_series=dataset_series,
                        availability_status=availability_status,
                        availability_message=availability_message,
                    )
                    continue
                self._iter_over_series(
                    resolved_model_spec, model, dataset_series)
            finally:
                self.progress.model_finished()

    def _iter_over_series(
            self,
            model_spec: ModelSpec,
            model: ForecastingModelAdapter,
            dataset_series: tuple[ForecastingSeriesRecord, ...],
    ) -> None:
        for series_record in dataset_series:
            item_key = self.incremental_persistence.item_key(
                series_record, model.name)
            if item_key in self.resumed_item_keys:
                existing_record = next(
                    (
                        record for record in self.run_records
                        if record.benchmark == series_record.benchmark
                        and record.dataset_name == series_record.dataset_name
                        and record.subset == series_record.subset
                        and record.series_id == series_record.series_id
                        and record.model_name == model.name
                    ),
                    None,
                )
                self.progress.item_resumed(
                    series_record.dataset_name,
                    model.name,
                    series_record.series_id,
                    existing_record.status.value if existing_record is not None else 'success',
                )
                continue
            self.progress.item_started(
                series_record.dataset_name, model.name, series_record.series_id)
            regime_diagnostics, routing_recommendation = self._build_series_context(
                series_record)
            try:
                self._evaluate_series(
                    model_spec, model, series_record, regime_diagnostics, routing_recommendation)
                self.progress.advance(RunStatus.SUCCESS.value)
            except ModelExecutionError as exc:
                self._append_failed_run_record(
                    model_spec=model_spec,
                    model=model,
                    series_record=series_record,
                    status=exc.status,
                    message=exc.message,
                    regime_diagnostics=regime_diagnostics,
                    routing_recommendation=routing_recommendation,
                )
                self.progress.advance(exc.status.value, exc.message)
            except Exception as exc:
                self._append_failed_run_record(
                    model_spec=model_spec,
                    model=model,
                    series_record=series_record,
                    status=RunStatus.FAILED,
                    message=str(exc),
                    regime_diagnostics=regime_diagnostics,
                    routing_recommendation=routing_recommendation,
                )
                self.progress.advance(RunStatus.FAILED.value, str(exc))

    def _handle_unavailable_model(
            self,
            *,
            model_spec: ModelSpec,
            model: ForecastingModelAdapter,
            dataset_series: tuple[ForecastingSeriesRecord, ...],
            availability_status: RunStatus,
            availability_message: str,
    ) -> None:
        for series_record in dataset_series:
            item_key = self.incremental_persistence.item_key(
                series_record, model.name)
            if item_key in self.resumed_item_keys:
                existing_record = next(
                    (
                        record for record in self.run_records
                        if record.benchmark == series_record.benchmark
                        and record.dataset_name == series_record.dataset_name
                        and record.subset == series_record.subset
                        and record.series_id == series_record.series_id
                        and record.model_name == model.name
                    ),
                    None,
                )
                self.progress.item_resumed(
                    series_record.dataset_name,
                    model.name,
                    series_record.series_id,
                    existing_record.status.value if existing_record is not None else availability_status.value,
                )
                continue
            self.progress.item_started(
                series_record.dataset_name, model.name, series_record.series_id)
            regime_diagnostics, routing_recommendation = self._build_series_context(
                series_record)
            self.run_records.append(
                BenchmarkRunRecord(
                    run_id=self.run_id,
                    benchmark=series_record.benchmark,
                    dataset_name=series_record.dataset_name,
                    subset=series_record.subset,
                    series_id=series_record.series_id,
                    model_name=model.name,
                    status=availability_status,
                    tags=model.tags,
                    message=availability_message,
                    metadata=self._build_common_metadata(
                        model_spec=model_spec,
                        model=model,
                        regime_diagnostics=regime_diagnostics,
                        routing_recommendation=routing_recommendation,
                        extra={'optional': model.optional},
                    ),
                )
            )
            self.incremental_persistence.persist_item_result(
                series_record=series_record,
                run_record=self.run_records[-1],
            )
            self.progress.advance(
                availability_status.value, availability_message)

    def _build_series_context(self, series_record: ForecastingSeriesRecord):
        regime_diagnostics = analyze_regime_diagnostics(
            np.asarray(series_record.train_values, dtype=float))
        routing_recommendation = recommend_forecasting_model(
            regime_diagnostics)
        return regime_diagnostics, routing_recommendation

    def _build_common_metadata(
            self,
            *,
            model_spec: ModelSpec,
            model: ForecastingModelAdapter,
            regime_diagnostics,
            routing_recommendation,
            extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            'adapter_name': model_spec.adapter_name,
            'model_adapter_family': adapter_name_to_family(model_spec.adapter_name),
            'regime_diagnostics': regime_diagnostics.to_dict(),
            'routing_recommendation': routing_recommendation.to_dict(),
            'routing_recommendation_family': adapter_name_to_family(routing_recommendation.primary_adapter),
            **self._runner_context_metadata(),
            **dict(extra or {}),
        }

    def _validate_forecast_length(
            self,
            series_record: ForecastingSeriesRecord,
            prediction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.artifacts_recorder.validate_forecast_length(series_record, prediction)

    def _record_metric_bundle(
            self,
            *,
            series_record: ForecastingSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            forecast: np.ndarray,
            metadata: dict[str, Any],
            metric_name_suffix: str = '',
    ) -> tuple[dict[str, float], dict[str, float]]:
        return self.artifacts_recorder.record_metric_bundle(
            series_record=series_record,
            model_name=model_name,
            actual=actual,
            forecast=forecast,
            metadata=metadata,
            metric_name_suffix=metric_name_suffix,
        )

    def _record_predictions(
            self,
            *,
            series_record: ForecastingSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            forecast: np.ndarray,
    ) -> None:
        self.artifacts_recorder.record_predictions(
            series_record=series_record,
            model_name=model_name,
            actual=actual,
            forecast=forecast,
        )

    def _build_tuned_model_spec(self, model_spec: ModelSpec, best_parameters: dict[str, Any]) -> ModelSpec:
        return self.post_fit_tuning._build_tuned_model_spec(model_spec, best_parameters)

    def _resolve_post_fit_tuning_runtime_config(
            self,
            model_spec: ModelSpec,
            baseline_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return self.post_fit_tuning._resolve_runtime_config(model_spec, baseline_metadata)

    def _maybe_run_post_fit_tuning_comparison(
            self,
            *,
            model_spec: ModelSpec,
            model: ForecastingModelAdapter,
            series_record: ForecastingSeriesRecord,
            baseline_metadata: dict[str, Any],
            baseline_metrics_summary: dict[str, float],
            regime_diagnostics,
            routing_recommendation,
    ) -> dict[str, Any]:
        return self.post_fit_tuning.run(
            model_spec=model_spec,
            model=model,
            series_record=series_record,
            baseline_metadata=baseline_metadata,
            baseline_metrics_summary=baseline_metrics_summary,
            regime_diagnostics=regime_diagnostics,
            routing_recommendation=routing_recommendation,
        )

    def _append_failed_run_record(
            self,
            *,
            model_spec: ModelSpec,
            model: ForecastingModelAdapter,
            series_record: ForecastingSeriesRecord,
            status: RunStatus,
            message: str,
            regime_diagnostics,
            routing_recommendation,
    ) -> BenchmarkRunRecord:
        run_record = BenchmarkRunRecord(
            run_id=self.run_id,
            benchmark=series_record.benchmark,
            dataset_name=series_record.dataset_name,
            subset=series_record.subset,
            series_id=series_record.series_id,
            model_name=model.name,
            status=status,
            tags=model.tags,
            message=message,
            metadata=self._build_common_metadata(
                model_spec=model_spec,
                model=model,
                regime_diagnostics=regime_diagnostics,
                routing_recommendation=routing_recommendation,
                extra={'optional': model.optional},
            ),
        )
        self.run_records.append(run_record)
        self.incremental_persistence.persist_item_result(
            series_record=series_record,
            run_record=run_record,
        )
        return run_record

    def _evaluate_series(
            self,
            model_spec: ModelSpec,
            model: ForecastingModelAdapter,
            series_record: ForecastingSeriesRecord,
            regime_diagnostics,
            routing_recommendation,
    ) -> None:
        metric_count_before = len(self.metric_records)
        prediction_count_before = len(self.prediction_records)
        quantile_prediction_count_before = len(
            self.quantile_prediction_records)
        # prediction, metadata = model.forecast(series_record)
        # actual, forecast = self._validate_forecast_length(series_record, prediction)

        scenario = self.config.scenario_spec

        if scenario is not None:
            if scenario.run_mode == RunMode.ZERO_SHOT:
                series_record = dataclasses.replace(
                    series_record, train_values=())
            elif scenario.run_mode == RunMode.FEW_SHOT:
                train = series_record.train_values
                if train:
                    n = len(train)
                    k = max(3, min(n // 5, 50))
                    series_record = dataclasses.replace(
                        series_record, train_values=train[-k:])

        raw_prediction = model.forecast(series_record)
        forecast_result = coerce_forecast_result(raw_prediction)
        validate_forecast_result_shapes(
            forecast_result, horizon=len(series_record.test_values))
        point_prediction, metadata = resolve_point_forecast(forecast_result)
        actual, forecast = self._validate_forecast_length(
            series_record, point_prediction)

        metrics_summary, event_metrics = self._record_metric_bundle(
            series_record=series_record,
            model_name=model.name,
            actual=actual,
            forecast=forecast,
            metadata=metadata,
        )
        self._record_predictions(
            series_record=series_record,
            model_name=model.name,
            actual=actual,
            forecast=forecast,
        )

        self.artifacts_recorder.record_quantile_predictions(
            series_record=series_record,
            model_name=model.name,
            actual=actual,
            forecast_result=forecast_result,
        )

        metadata = self._maybe_run_post_fit_tuning_comparison(
            model_spec=model_spec,
            model=model,
            series_record=series_record,
            baseline_metadata=metadata,
            baseline_metrics_summary=metrics_summary,
            regime_diagnostics=regime_diagnostics,
            routing_recommendation=routing_recommendation,
        )
        run_record = BenchmarkRunRecord(
            run_id=self.run_id,
            benchmark=series_record.benchmark,
            dataset_name=series_record.dataset_name,
            subset=series_record.subset,
            series_id=series_record.series_id,
            model_name=model.name,
            family=family.name,
            status=RunStatus.SUCCESS,
            tags=model.tags,
            metrics_summary=metrics_summary,
            metadata=self._build_common_metadata(
                model_spec=model_spec,
                model=model,
                regime_diagnostics=regime_diagnostics,
                routing_recommendation=routing_recommendation,
                extra={
                    **metadata,
                    'active_forecast_steps': int(event_metrics.get('active_forecast_steps', 0)),
                    'calm_forecast_steps': int(event_metrics.get('calm_forecast_steps', 0)),
                    'forecast_result_kind': describe_forecast_result_kind(forecast_result),
                },
            ),
        )
        self.run_records.append(run_record)
        self.incremental_persistence.persist_item_result(
            series_record=series_record,
            run_record=run_record,
            metric_records=tuple(self.metric_records[metric_count_before:]),
            prediction_records=tuple(
                self.prediction_records[prediction_count_before:]),
            quantile_prediction_records=tuple(
                self.quantile_prediction_records[quantile_prediction_count_before:]),
        )


def run_forecasting_suite(config: BenchmarkSuiteConfig) -> ForecastingBenchmarkResult:
    """Compatibility entrypoint that executes a ForecastingSuiteRunner."""
    return ForecastingSuiteRunner(config).run_suite()

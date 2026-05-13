from __future__ import annotations

import importlib
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

from fedot_ind.core.repository.detection_registry import canonical_detection_model_name

from benchmark.v2.core import (
    BenchmarkSuiteConfig,
    DetectionSeriesRecord,
    DetectionModelAdapter,
    DatasetSpec,
    MetricRecord,
    ModelSpec,
    PredictionRecord,
    RunStatus,
    TaskType,
)


SUPPORTED_ANOMALY_DETECTION_METRICS = ('accuracy', 'balanced_accuracy', 'f1_macro')


class BenchmarkConfigurationError(ValueError):
    pass


def validate_detection_suite_config(config: BenchmarkSuiteConfig) -> None:
    """проверка, что BenchmarkSuiteConfig подходит для задачи anomaly_detection"""
    if config.task_type is not TaskType.ANOMALY_DETECTION:
        raise BenchmarkConfigurationError('Anomaly detection suite expects task_type=anomaly_detection.')
    if not config.datasets:
        raise BenchmarkConfigurationError('Benchmark suite must contain at least one dataset spec.')
    if not config.models:
        raise BenchmarkConfigurationError('Benchmark suite must contain at least one model spec.')
    unsupported = set(config.metrics) - set(SUPPORTED_ANOMALY_DETECTION_METRICS)
    if unsupported:
        raise BenchmarkConfigurationError(f'Unsupported anomaly detection metrics: {sorted(unsupported)}')


class SKABAdapter:
    """чтение файла датасета skab и приведение к единому формату.
    Адапрер возвращает record (один временной ряд, у которого есть train/test и horizon).
    После adapter-а runner работает одинаково с любым датасетом.
    """
    benchmark_name = 'skab'

    def load_series(self, spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        ...


class MPSIAdapter:
    """чтение файла датасета mpsi и приведение к единому формату.
    Адапрер возвращает record (один временной ряд, у которого есть train/test и horizon).
    После adapter-а runner работает одинаково с любым датасетом.
    """
    benchmark_name = 'mpsi'

    def load_series(self, spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        ...


def build_dataset_adapter(spec: DatasetSpec):
    """выбор адаптера"""
    benchmark = spec.benchmark.lower()
    if benchmark == 'skab':
        return SKABAdapter()
    if benchmark == 'mpsi':
        return MPSIAdapter()
    # if benchmark == 'in_memory':
    #     return InMemoryDetectionAdapter() TODO: разобраться что такое InMemoryDetectionAdapter в forecasting
    raise BenchmarkConfigurationError(...)


def build_model_adapter(spec: ModelSpec) -> DetectionModelAdapter: # TODO: реализовать классы моделей, наследуемые от DetectionModelAdapter
    """выбор модели. создаёт объект класса модели с 
    аргументами для работы модели и функциями availability() и forecast()"""
    raw_adapter_name = spec.adapter_name.lower()
    adapter_name = canonical_detection_model_name(raw_adapter_name)
    params = dict(spec.params)

    # if adapter_name in CANONICAL_STAGE_DETECTION_MODELS:
    #     return RuntimeDetectionModelAdapter(...)

    # raise BenchmarkConfigurationError(...)
    if raw_adapter_name == 'okhs':
        return OKHSModel(name=spec.display_name, tags=spec.tags or ('okhs', 'forecasting'), **params)
    if adapter_name == 'okhs_fdmd_forecaster':
        return OKHSFDMDForecasterModel(
            name=spec.display_name,
            tags=spec.tags or ('okhs', 'forecasting', 'operator_model'),
            **params,
        )
 
    if adapter_name in {'mssa', 'mssa_forecaster'}:
        return MSSAModel(name=spec.display_name, tags=spec.tags or ('baseline', 'forecasting', 'mssa'), **params)
    
    if adapter_name == 'nbeats':
        return OptionalExternalModel(
            dependency_name='neuralforecast',
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'forecasting', 'external', 'nbeats'),
        )
    if adapter_name == 'tft':
        return OptionalExternalModel(
            dependency_name='pytorch_forecasting',
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'forecasting', 'external', 'tft'),
        )
    raise BenchmarkConfigurationError(f'Unsupported forecasting model adapter: {spec.adapter_name}')

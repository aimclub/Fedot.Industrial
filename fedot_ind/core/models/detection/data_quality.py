from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fedot_ind.core.models.detection.runtime import ensure_detection_array


# Timestamp alignment

def align_timestamps(
        values: np.ndarray,
        timestamps: Sequence[str | float] | None,
        *,
        target_sample_rate_hz: float | None = None,
        duplicate_policy: str = 'keep_last',   # 'keep_last' | 'mean' | 'drop'
        gap_policy: str = 'mark_only',          # 'mark_only' | 'forward_fill' | 'interpolate_linear'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Выравнивает ряд по временной оси (равномерные интервалы между отсчётами): сортировка,
    дедупликация, диагностика/заполнение пропусков.

    Возвращает кортеж (aligned_values, aligned_timestamps, gap_mask, alignment_report):
        aligned_values     : np.ndarray (n_aligned, n_channels)
        aligned_timestamps : np.ndarray (n_aligned,) в секундах
        gap_mask           : np.ndarray (n_aligned,) bool, True = точка недостоверна
                             (синтетически заполнена либо NaN)
        alignment_report   : dict с n_duplicates, n_gaps, gap_total_seconds,
                             resampling_method, original_n, aligned_n, ...

    Ресемплинг на регулярную сетку выполняется если задан target_sample_rate_hz.
    Иначе ряд лишь сортируется и дедуплицируется, а gap-статистика считается для отчёта.
    """
    # вход стал (n_samples, n_channels)
    series = ensure_detection_array(values)
    original_n = int(series.shape[0])  # фиктивная регулярная ось

    if timestamps is None:
        # временных меток нет -> регулярный целочисленный индекс; маскируем только NaN
        gap_mask = np.isnan(series).any(axis=1)
        report = {
            'n_duplicates': 0,
            'n_gaps': 0,
            'gap_total_seconds': 0.0,
            'resampling_method': 'none',
            'original_n': original_n,
            'aligned_n': original_n,
            'has_timestamps': False,
            'nominal_dt_seconds': 0.0,
        }
        return series, np.arange(original_n, dtype=float), gap_mask, report

    seconds = _timestamps_to_seconds(timestamps)
    if seconds.shape[0] != original_n:
        raise ValueError('timestamps length must match the number of samples in values.')

    order = np.argsort(seconds, kind='stable')
    seconds = seconds[order]
    series = series[order]
    # дедупликация
    series, seconds, n_duplicates = _resolve_duplicates(series, seconds, duplicate_policy)

    diffs = np.diff(seconds)
    positive = diffs[diffs > 0]
    if target_sample_rate_hz is not None and float(target_sample_rate_hz) > 0:
        nominal_dt = 1.0 / float(target_sample_rate_hz)
    elif positive.size:
        nominal_dt = float(np.median(positive))
    else:
        nominal_dt = 0.0

    if nominal_dt > 0 and diffs.size:
        gap_factor = diffs / nominal_dt
        gap_positions = gap_factor > 1.5
        n_gaps = int(np.count_nonzero(gap_positions))
        gap_total_seconds = float(np.sum(diffs[gap_positions] - nominal_dt)) if n_gaps else 0.0
    else:
        n_gaps = 0
        gap_total_seconds = 0.0

    if target_sample_rate_hz is not None and nominal_dt > 0 and seconds.size >= 2:
        aligned_values, aligned_seconds, gap_mask, method = _resample_to_grid(
            series, seconds, nominal_dt, gap_policy,
        )
    else:
        aligned_values, aligned_seconds = series, seconds
        gap_mask = np.isnan(series).any(axis=1)
        method = 'none'

    report = {
        'n_duplicates': int(n_duplicates),
        'n_gaps': int(n_gaps),
        'gap_total_seconds': float(gap_total_seconds),
        'resampling_method': method,
        'original_n': original_n,
        'aligned_n': int(aligned_values.shape[0]),
        'has_timestamps': True,
        'nominal_dt_seconds': float(nominal_dt),
    }
    return aligned_values, aligned_seconds, gap_mask, report


def _timestamps_to_seconds(timestamps: Sequence[str | float]) -> np.ndarray:
    """Приводит метки времени к секундам"""
    array = np.asarray(list(timestamps))
    if array.dtype.kind in {'i', 'u', 'f'}:
        return array.astype(float)
    try:
        return array.astype(float)  # числовые строки '0.0', '1.0', ...
    except (ValueError, TypeError):
        # TODO: pandas используется только ради парсинга ISO-строк в pd.to_datetime
        parsed = pd.DatetimeIndex(pd.to_datetime(array, errors='coerce'))
        if parsed.isna().any():
            raise ValueError(
                'Could not parse some timestamps; pass numeric seconds or ISO datetime strings.'
            )
        return parsed.astype('int64').to_numpy(dtype=float) / 1e9


def _resolve_duplicates(
        series: np.ndarray,
        seconds: np.ndarray,
        policy: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Снимает дубликаты по одинаковым timestamp согласно policy."""
    if seconds.size == 0:
        return series, seconds, 0
    unique_t, inverse, counts = np.unique(seconds, return_inverse=True, return_counts=True)
    n_duplicates = int(seconds.size - unique_t.size)
    if n_duplicates == 0:
        return series, seconds, 0
    normalized = str(policy).lower()
    if normalized == 'mean':
        aggregated = np.zeros((unique_t.size, series.shape[1]), dtype=float)
        np.add.at(aggregated, inverse, series)
        aggregated /= counts[:, None]
        return aggregated, unique_t, n_duplicates
    if normalized == 'drop':
        keep = counts[inverse] == 1
        return series[keep], seconds[keep], int(np.count_nonzero(~keep))
    # keep_last для каждого уникального времени берём последнее вхождение
    last_index = np.empty(unique_t.size, dtype=int)
    last_index[inverse] = np.arange(inverse.size)
    return series[last_index], unique_t, n_duplicates


def _resample_to_grid(
        series: np.ndarray,
        seconds: np.ndarray,
        nominal_dt: float,
        gap_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Перенос наблюдения на регулярную сетку и заполняет пропуски по gap_policy."""
    grid = np.arange(seconds[0], seconds[-1] + 0.5 * nominal_dt, nominal_dt)
    if grid.size == 0:
        grid = seconds.copy()
    aligned = np.full((grid.size, series.shape[1]), np.nan, dtype=float)
    position = np.clip(np.round((seconds - seconds[0]) / nominal_dt).astype(int), 0, grid.size - 1)
    aligned[position] = series
    observed = np.zeros(grid.size, dtype=bool)
    observed[position] = True
    gap_mask = ~observed

    normalized = str(gap_policy).lower()
    if normalized == 'forward_fill':
        aligned = _forward_fill(aligned)
        method = 'forward_fill'
    elif normalized == 'interpolate_linear':
        aligned = _interpolate_linear(aligned)
        method = 'interpolate_linear'
    else:
        method = 'mark_only'

    gap_mask = gap_mask | np.isnan(aligned).any(axis=1)
    return aligned, grid, gap_mask, method


def _forward_fill(values: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for channel in range(filled.shape[1]):
        column = filled[:, channel]
        valid = ~np.isnan(column)
        if not valid.any():
            continue
        carry = np.where(valid, np.arange(column.size), 0)
        np.maximum.accumulate(carry, out=carry)
        column = column[carry]
        first_valid = int(np.argmax(valid))
        column[:first_valid] = column[first_valid]  # backfill ведущих NaN
        filled[:, channel] = column
    return filled


def _interpolate_linear(values: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for channel in range(filled.shape[1]):
        column = filled[:, channel]
        valid_idx = np.where(~np.isnan(column))[0]
        if valid_idx.size == 0:
            continue
        filled[:, channel] = np.interp(np.arange(column.size), valid_idx, column[valid_idx])
    return filled


# Channel-quality diagnostics
@dataclass(frozen=True)
class ChannelQualityReport:
    channel_name: str
    n_missing: int
    n_duplicates: int
    is_dead: bool
    saturation_segments: tuple[tuple[int, int], ...]
    dropout_segments: tuple[tuple[int, int], ...]
    noise_floor: float
    dynamic_range: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'channel_name': self.channel_name,
            'n_missing': int(self.n_missing),
            'n_duplicates': int(self.n_duplicates),
            'is_dead': bool(self.is_dead),
            'saturation_segments': [list(segment) for segment in self.saturation_segments],
            'dropout_segments': [list(segment) for segment in self.dropout_segments],
            'noise_floor': float(self.noise_floor),
            'dynamic_range': float(self.dynamic_range),
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class DataQualityReport:
    n_channels: int
    n_samples: int
    channel_reports: tuple[ChannelQualityReport, ...]
    overall_pct_missing: float
    has_dead_channels: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dead_channel_names(self) -> tuple[str, ...]:
        return tuple(report.channel_name for report in self.channel_reports if report.is_dead)

    def to_dict(self) -> dict[str, Any]:
        return {
            'n_channels': int(self.n_channels),
            'n_samples': int(self.n_samples),
            'channel_reports': [report.to_dict() for report in self.channel_reports],
            'overall_pct_missing': float(self.overall_pct_missing),
            'has_dead_channels': bool(self.has_dead_channels),
            'dead_channel_names': list(self.dead_channel_names),
            **dict(self.metadata),
        }


def diagnose_channel_quality(
        values: np.ndarray,
        channel_names: Sequence[str] | None = None,
        *,
        dead_var_eps: float = 1e-10,
        saturation_quantile: float = 0.99,
        min_saturation_length: int = 10,
) -> DataQualityReport:
    """
    поканальный (per-channel) анализ качества сигналов
    диагностика: dead channels, насыщение, дропауты, noise floor, динамический диапазон"""
    series = ensure_detection_array(values)
    n_samples, n_channels = series.shape
    names = tuple(channel_names) if channel_names else tuple(f'channel_{i}' for i in range(n_channels))
    if len(names) != n_channels:
        raise ValueError('channel_names length must match the number of channels.')

    reports = [
        _diagnose_single_channel(
            series[:, channel], names[channel],
            dead_var_eps=dead_var_eps,
            saturation_quantile=saturation_quantile,
            min_saturation_length=min_saturation_length,
        )
        for channel in range(n_channels)
    ]
    total_missing = sum(report.n_missing for report in reports)
    overall_pct_missing = 100.0 * float(total_missing) / float(max(1, n_samples * n_channels))
    return DataQualityReport(
        n_channels=int(n_channels),
        n_samples=int(n_samples),
        channel_reports=tuple(reports),
        overall_pct_missing=overall_pct_missing,
        has_dead_channels=any(report.is_dead for report in reports),
    )


def _diagnose_single_channel(
        column: np.ndarray,
        name: str,
        *,
        dead_var_eps: float,
        saturation_quantile: float,
        min_saturation_length: int,
) -> ChannelQualityReport:
    column = np.asarray(column, dtype=float)
    n_missing = int(np.count_nonzero(np.isnan(column)))
    finite = column[~np.isnan(column)]
    if finite.size == 0:
        return ChannelQualityReport(
            channel_name=name, n_missing=n_missing, n_duplicates=0, is_dead=True,
            saturation_segments=(), dropout_segments=(), noise_floor=0.0, dynamic_range=0.0,
            metadata={'all_missing': True},
        )

    is_dead = float(np.var(finite)) < float(dead_var_eps)

    same_as_prev = np.zeros(column.size, dtype=bool)
    same_as_prev[1:] = (
        (column[1:] == column[:-1]) & ~np.isnan(column[1:]) & ~np.isnan(column[:-1])
    )
    n_duplicates = int(np.count_nonzero(same_as_prev))

    high = float(np.quantile(finite, saturation_quantile))
    low = float(np.quantile(finite, 1.0 - saturation_quantile))
    saturation_flags = np.where(np.isnan(column), False, (column >= high) | (column <= low))
    saturation_segments = _contiguous_segments(saturation_flags, min_saturation_length)

    dropout_flags = same_as_prev | np.isnan(column)
    dropout_segments = _contiguous_segments(dropout_flags, min_saturation_length)

    diffs = np.abs(np.diff(finite))
    noise_floor = float(np.median(diffs)) if diffs.size else 0.0
    dynamic_range = float(np.max(finite) - np.min(finite))

    return ChannelQualityReport(
        channel_name=name,
        n_missing=n_missing,
        n_duplicates=n_duplicates,
        is_dead=bool(is_dead),
        saturation_segments=saturation_segments,
        dropout_segments=dropout_segments,
        noise_floor=noise_floor,
        dynamic_range=dynamic_range,
    )


def _contiguous_segments(flags: np.ndarray, min_length: int) -> tuple[tuple[int, int], ...]:
    """Склеивает True-флаги в непрерывные сегменты [start, end] длиной >= min_length."""
    segments: list[tuple[int, int]] = []
    start = None
    size = int(flags.size)
    minimum = max(1, int(min_length))
    for index in range(size):
        if flags[index] and start is None:
            start = index
        elif not flags[index] and start is not None:
            if index - start >= minimum:
                segments.append((int(start), int(index - 1)))
            start = None
    if start is not None and size - start >= minimum:
        segments.append((int(start), int(size - 1)))
    return tuple(segments)


def prepare_detection_series(
        values: Sequence[float] | np.ndarray,
        timestamps: Sequence[str | float] | None = None,
        *,
        duplicate_policy: str = 'keep_last',
        gap_policy: str = 'mark_only',
        target_sample_rate_hz: float | None = None,
        channel_names: Sequence[str] | None = None,
        run_channel_quality: bool = True,
        dead_var_eps: float = 1e-10,
        saturation_quantile: float = 0.99,
        min_saturation_length: int = 10,
) -> tuple[np.ndarray, np.ndarray, DataQualityReport]:
    """alignment + gap_mask + (опц.) channel-quality.
    Raw data
    ↓
    align_timestamps
    ↓
    Get an aligned row
    ↓
    diagnose_channel_quality
    ↓
    Get a quality report
    ↓
    Return everything together
"""
    aligned_values, _aligned_ts, gap_mask, alignment_report = align_timestamps(
        values,
        timestamps,
        target_sample_rate_hz=target_sample_rate_hz,
        duplicate_policy=duplicate_policy,
        gap_policy=gap_policy,
    )
    gap_mask = np.asarray(gap_mask, dtype=bool).reshape(-1)

    if run_channel_quality:
        quality_report = diagnose_channel_quality(
            aligned_values,
            channel_names=channel_names,
            dead_var_eps=dead_var_eps,
            saturation_quantile=saturation_quantile,
            min_saturation_length=min_saturation_length,
        )
    else:
        n_samples, n_channels = ensure_detection_array(aligned_values).shape
        quality_report = DataQualityReport(
            n_channels=int(n_channels),
            n_samples=int(n_samples),
            channel_reports=(),
            overall_pct_missing=0.0,
            has_dead_channels=False,
        )

    quality_report = DataQualityReport(
        n_channels=quality_report.n_channels,
        n_samples=quality_report.n_samples,
        channel_reports=quality_report.channel_reports,
        overall_pct_missing=quality_report.overall_pct_missing,
        has_dead_channels=quality_report.has_dead_channels,
        metadata={
            **dict(quality_report.metadata),
            'alignment_report': alignment_report,
            'duplicate_policy': str(duplicate_policy),
            'gap_policy': str(gap_policy),
            'target_sample_rate_hz': target_sample_rate_hz,
        },
    )
    return aligned_values, gap_mask, quality_report

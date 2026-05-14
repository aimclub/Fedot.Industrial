from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - lightweight environments may not have FEDOT
    from fedot.core.data.data import InputData
except Exception:  # pragma: no cover
    InputData = None


class DetectionSplitKind(str, Enum):
    HOLDOUT = 'holdout'
    TEMPORAL = 'temporal'
    DOMAIN_HOLDOUT = 'domain_holdout'


@dataclass(frozen=True)
class DetectionSplitSpec:
    kind: DetectionSplitKind = DetectionSplitKind.HOLDOUT
    train_fraction: float = 0.7
    calibration_fraction: float = 0.15
    random_seed: int = 0
    prevent_future_leakage: bool = True
    target_domain: str | None = None


@dataclass(frozen=True)
class DetectionWindowBatch:
    windows: np.ndarray
    window_indices: np.ndarray
    original_length: int
    window_size: int
    stride: int
    channel_names: tuple[str, ...] = ()
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return int(self.windows.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.windows.shape[2])

    @property
    def flattened_features(self) -> np.ndarray:
        return self.windows.reshape(self.n_windows, -1)

    @property
    def statistical_features(self) -> np.ndarray:
        return build_window_statistical_features(self.windows)

    def to_dict(self) -> dict[str, Any]:
        return {
            'windows_shape': tuple(int(value) for value in self.windows.shape),
            'window_indices_shape': tuple(int(value) for value in self.window_indices.shape),
            'original_length': int(self.original_length),
            'window_size': int(self.window_size),
            'stride': int(self.stride),
            'channel_names': list(self.channel_names),
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class RegimeSegment:
    start_index: int
    end_index: int
    regime_label: str
    mean_level: float
    volatility: float
    slope: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnomalyScoreSeries:
    scores: tuple[float, ...]
    labels: tuple[int, ...]
    threshold: float
    calibration_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'scores': list(self.scores),
            'labels': list(self.labels),
            'threshold': float(self.threshold),
            'calibration_strategy': self.calibration_strategy,
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class DetectionEvent:
    start_index: int
    end_index: int
    peak_index: int
    peak_score: float
    mean_score: float
    label: str = 'anomaly'
    regime_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransferAlignmentReport:
    strategy: str
    source_domain: str
    target_domain: str
    n_source: int
    n_target: int
    source_channel_mean: tuple[float, ...]
    target_channel_mean: tuple[float, ...]
    mean_shift: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'strategy': self.strategy,
            'source_domain': self.source_domain,
            'target_domain': self.target_domain,
            'n_source': int(self.n_source),
            'n_target': int(self.n_target),
            'source_channel_mean': list(self.source_channel_mean),
            'target_channel_mean': list(self.target_channel_mean),
            'mean_shift': list(self.mean_shift),
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class DetectionSeriesEvaluation:
    model_name: str
    canonical_model_name: str
    family: str
    parameters: dict[str, Any]
    primary_metric: str
    metrics: dict[str, float]
    event_metrics: dict[str, float]
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'canonical_model_name': self.canonical_model_name,
            'family': self.family,
            'parameters': dict(self.parameters),
            'primary_metric': self.primary_metric,
            'metrics': {str(key): float(value) for key, value in self.metrics.items()},
            'event_metrics': {str(key): float(value) for key, value in self.event_metrics.items()},
            'threshold': float(self.threshold),
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class RiskFeatureFrame:
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows), columns=list(self.columns))

    def to_dict(self) -> dict[str, Any]:
        return {
            'columns': list(self.columns),
            'rows': list(self.rows),
            **dict(self.metadata),
        }


class DetectionBoundaryAdapter:
    @staticmethod
    def from_input_data(
            input_data: InputData,
            *,
            window_size: int | None = None,
            window_size_percent: float | None = None,
            stride: int | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> DetectionWindowBatch:
        if InputData is None:  # pragma: no cover
            raise ValueError('FEDOT InputData is unavailable in the current environment.')
        values = ensure_detection_array(input_data.features)
        resolved_window_size = resolve_detection_window_size(
            values.shape[0],
            window_size=window_size,
            window_size_percent=window_size_percent,
        )
        return build_detection_window_batch(
            values,
            window_size=resolved_window_size,
            stride=resolve_detection_stride(resolved_window_size, stride),
            metadata={'idx': getattr(input_data, 'idx', None), **dict(metadata or {})},
        )


def ensure_detection_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Преобразует входные данные в двумерный массив numpy с плавающей точкой, пригодный для детектирования.

    - Если входной массив одномерный, преобразует в столбец (n, 1).
    - Если двумерный — оставляет без изменений.
    - Если размерность больше 2 — сворачивает все оси, кроме первой, в одну.

    Параметры:
        values : Sequence[float] или np.ndarray
            Входные данные (список, кортеж, массив).

    Возвращает:
        np.ndarray
            Двумерный массив с shape (n_samples, n_features).

    Исключения:
        ValueError
            Если передан 0-мерный массив (скаляр).
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    if array.ndim > 2:
        leading = array.shape[0]
        return array.reshape(leading, -1)
    raise ValueError('Detection input must be at least one-dimensional.')


def resolve_detection_window_size(
        series_length: int,
        *,
        window_size: int | None = None,
        window_size_percent: float | None = None,
        minimum_window_size: int = 8,
) -> int:
    """
    Определяет размер окна для скользящего анализа временного ряда.

    Правила:
    1. Если длина ряда меньше 4 — возвращает длину ряда (минимум 1).
    2. Если задан window_size — возвращает max(2, min(window_size, series_length)).
    3. Если задан window_size_percent — вычисляет размер как процент от series_length,
       затем ограничивает max(2, min(..., series_length)).
    4. Иначе — default = max(minimum_window_size, round(series_length * 0.1)), затем
       ограничивает series_length.

    Параметры:
        series_length : int
            Общее количество точек ряда.
        window_size : int или None
            Явный размер окна.
        window_size_percent : float или None
            Размер окна в процентах от длины ряда.
        minimum_window_size : int, default=8
            Минимальный размер окна по умолчанию.

    Возвращает:
        int
            Итоговый размер окна (всегда >=2 для рядов длиной >=4).
    
    Важно:
    --
    Используется round() - надо узнать, насколько это важно для контекста
    """
    if series_length < 4:
        return max(1, int(series_length))
    if window_size is not None:
        return max(2, min(int(window_size), int(series_length)))
    if window_size_percent is not None:
        resolved = int(round(series_length * float(window_size_percent) / 100.0))
        return max(2, min(resolved, int(series_length)))
    default_window = max(minimum_window_size, int(round(series_length * 0.1)))
    return min(default_window, int(series_length))


def resolve_detection_stride(window_size: int, stride: int | None = None) -> int:
    """
    Определяет шаг сдвига окна (stride) для детектирования.

    - Если stride задан явно — возвращает max(1, int(stride)).
    - Иначе — возвращает max(1, window_size // 4).

    Параметры:
        window_size : int
            Размер окна.
        stride : int или None
            Шаг сдвига (если None — выбирается автоматически).

    Возвращает:
        int
            Шаг сдвига (всегда >= 1).
    """
    if stride is not None:
        return max(1, int(stride))
    return max(1, int(window_size // 4))


def build_detection_window_batch(
        values: Sequence[float] | np.ndarray,
        *,
        window_size: int,
        stride: int = 1,
        channel_names: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
) -> DetectionWindowBatch:
    """
    Нарезает временной ряд на окна заданного размера с заданным шагом.

    Создаёт объект (DetectionWindowBatch), содержащий:
    - трёхмерный массив окон (n_windows, window_size, n_channels),
    - индексы начала и конца каждого окна (n_windows, 2),
    - метаданные (длина ряда, размер окна, шаг, имена каналов и пользовательские метаданные).

    Параметры:
        values : Sequence[float] или np.ndarray
            Входной ряд (1D или 2D, где второй размер — каналы).
        window_size : int
            Размер окна.
        stride : int, default=1
            Шаг сдвига.
        channel_names : Sequence[str] или None
            Имена каналов. Если None — генерируются как 'channel_0', 'channel_1', ...
        metadata : dict или None
            Дополнительные метаданные.

    Возвращает:
        DetectionWindowBatch
            Сформированный пакет окон.

    Исключения:
        ValueError
            Если длина ряда меньше window_size.
    """
    series = ensure_detection_array(values)
    if series.shape[0] < window_size:
        raise ValueError(
            f'Series length {series.shape[0]} is shorter than the requested detection window {window_size}.'
        )
    windows = []
    indices = []
    for start in range(0, series.shape[0] - window_size + 1, stride):
        end = start + window_size
        windows.append(series[start:end])
        indices.append((start, end))
    window_tensor = np.asarray(windows, dtype=float)
    window_index = np.asarray(indices, dtype=int)
    return DetectionWindowBatch(
        windows=window_tensor,
        window_indices=window_index,
        original_length=int(series.shape[0]),
        window_size=int(window_size),
        stride=int(stride),
        channel_names=tuple(channel_names or tuple(f'channel_{index}' for index in range(series.shape[1]))),
        metadata=dict(metadata or {}),
    )


def split_detection_batch(
        batch: DetectionWindowBatch,
        split_spec: DetectionSplitSpec,
) -> tuple[DetectionWindowBatch, DetectionWindowBatch, DetectionWindowBatch | None]:
    """
    Разбивает пакет детекции на тренировочную, калибровочную и тестовую выборки согласно спецификации.

    Поддерживаются ТРИ типа разбиения:
    - DOMAIN_HOLDOUT: выделяет одно доменное значение как целевое (target_domain),
      остальные домены — в тренировку. Внутри целевого домена окна делятся на калибровку и тест 
      с учётом временного порядка.
    - HOLDOUT: случайное или временное разбиение всех окон с заданными долями.
    - есть ещё TEMPORAL - без доменов. train отдельно, калибровка и тест делятся, соблюдая временной порядок 
      с помощью метода _split_temporal_window_ids(). 
      
    Для предотвращения утечки будущего (prevent_future_leakage=True) гарантируется,
    что калибровочные окна предшествуют тестовым во времени.

    Параметры:
        batch : DetectionWindowBatch
            Пакет окон, полученный из build_detection_window_batch.
        split_spec : DetectionSplitSpec
            Объект спецификации разбиения.

    Возвращает:
        tuple[DetectionWindowBatch, DetectionWindowBatch, DetectionWindowBatch | None]
            (train_batch, calibration_batch, test_batch). Калибровочный пакет может совпадать  тренировочным, если калибровочных окон нет.

    Исключения:
        ValueError
            Если batch не содержит окон, отсутствуют необходимые метаданные для доменного разбиения,
            или разбиение привело к пустому тренировочному набору.
    
    Важно:
    ------
    Надо проверить во всём процессе, нужно ли убеждаться в наличии метаданных, как таковых.

    Есть ещё момент, что calibration_fraction (при HOLDOUT - от всего количества окон, 
    при DOMAIN_HOLDOUT - от части без train) 

    """
    n_windows = batch.n_windows
    if n_windows == 0:
        raise ValueError('Detection batch must contain at least one window.')

    def _select(window_ids: np.ndarray, split_name: str) -> DetectionWindowBatch | None:
        selected = np.asarray(window_ids, dtype=int).reshape(-1)
        if selected.size == 0:
            return None
        selected = np.unique(selected)
        base_metadata = dict(batch.metadata)
        if 'window_domains' in base_metadata:
            base_metadata['window_domains'] = [base_metadata['window_domains'][index] for index in selected.tolist()]
        if 'domains' in base_metadata:
            base_metadata['domains'] = [base_metadata['domains'][index] for index in selected.tolist()]
        metadata = {
            **base_metadata,
            'split_kind': split_spec.kind.value,
            'split_name': split_name,
            'selected_window_ids': selected.tolist(),
        }
        return DetectionWindowBatch(
            windows=batch.windows[selected],
            window_indices=batch.window_indices[selected],
            original_length=batch.original_length,
            window_size=batch.window_size,
            stride=batch.stride,
            channel_names=batch.channel_names,
            mask=batch.mask,
            metadata=metadata,
        )

    if split_spec.kind is DetectionSplitKind.DOMAIN_HOLDOUT:
        window_domains = batch.metadata.get('window_domains', batch.metadata.get('domains'))
        if window_domains is None:
            raise ValueError('Domain-holdout detection split requires batch.metadata["window_domains"].')
        window_domains = np.asarray(window_domains, dtype=object).reshape(-1)
        if window_domains.shape[0] != n_windows:
            raise ValueError('window_domains length must match the number of detection windows.')
        target_domain = split_spec.target_domain or str(window_domains[-1])
        train_ids = np.flatnonzero(window_domains != target_domain)
        holdout_ids = np.flatnonzero(window_domains == target_domain)
        if train_ids.size == 0 or holdout_ids.size == 0:
            raise ValueError('Domain-holdout split requires both source and target-domain windows.')
        n_calibration = max(1, int(round(holdout_ids.size * split_spec.calibration_fraction)))
        calibration_ids, test_ids = _split_temporal_window_ids(
            batch.window_indices,
            holdout_ids,
            calibration_size=n_calibration,
            prevent_future_leakage=split_spec.prevent_future_leakage,
        )
        train_batch = _select(train_ids, 'train')
        calibration_batch = _select(calibration_ids, 'calibration')
        test_batch = _select(test_ids, 'test')
        if train_batch is None:
            raise ValueError('Detection split produced an empty train batch.')
        return train_batch, calibration_batch or train_batch, test_batch

    n_train = max(1, int(round(n_windows * split_spec.train_fraction)))
    remaining = max(0, n_windows - n_train)
    n_calibration = min(
        remaining,
        max(0, int(round(n_windows * split_spec.calibration_fraction))),
    )
    all_ids = np.arange(n_windows, dtype=int)

    if split_spec.kind is DetectionSplitKind.HOLDOUT and not split_spec.prevent_future_leakage:
        rng = np.random.RandomState(split_spec.random_seed)
        shuffled_ids = rng.permutation(all_ids)
        train_ids = np.sort(shuffled_ids[:n_train])
        calibration_ids = np.sort(shuffled_ids[n_train:n_train + n_calibration])
        test_ids = np.sort(shuffled_ids[n_train + n_calibration:])
    else:
        train_ids = all_ids[:n_train]
        remaining_ids = all_ids[n_train:]
        calibration_ids, test_ids = _split_temporal_window_ids(
            batch.window_indices,
            remaining_ids,
            calibration_size=n_calibration,
            prevent_future_leakage=split_spec.prevent_future_leakage,
            minimum_start=batch.window_indices[train_ids[-1], 1] if train_ids.size else None,
        )

    train_batch = _select(train_ids, 'train')
    calibration_batch = _select(calibration_ids, 'calibration')
    test_batch = _select(test_ids, 'test')
    if train_batch is None:
        raise ValueError('Detection split produced an empty train batch.')
    return train_batch, calibration_batch or train_batch, test_batch


def _split_temporal_window_ids(
        window_indices: np.ndarray,
        candidate_ids: np.ndarray,
        *,
        calibration_size: int,
        prevent_future_leakage: bool,
        minimum_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вспомогательная функция: разбивает идентификаторы окон на калибровочные и тестовые,
    соблюдая временной порядок (если prevent_future_leakage=True).

    Параметры:
        window_indices : np.ndarray
            Массив формы (n_windows, 2) с началом и концом каждого окна.
        candidate_ids : np.ndarray
            Массив идентификаторов окон, которые нужно разделить.
        calibration_size : int
            Сколько окон выделить в калибровку (берутся первые по времени).
        prevent_future_leakage : bool
            Если True, тестовые окна не могут начинаться раньше, чем заканчивается последнее калибровочное.
        minimum_start : int или None
            Минимальное значение start индекса для включения окна (фильтр).

    Возвращает:
        tuple[np.ndarray, np.ndarray]
            (calibration_ids, test_ids) – отсортированные массивы идентификаторов.
    """   
    candidates = np.asarray(candidate_ids, dtype=int).reshape(-1)
    if candidates.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    ordered = np.sort(candidates)
    if minimum_start is not None:
        ordered = ordered[window_indices[ordered, 0] >= int(minimum_start)]
    if ordered.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    calibration_size = max(0, int(calibration_size))
    calibration_ids = ordered[:calibration_size]
    if not prevent_future_leakage or calibration_ids.size == 0:
        return calibration_ids, ordered[calibration_size:]

    calibration_end = int(window_indices[calibration_ids[-1], 1])
    test_ids = ordered[window_indices[ordered, 0] >= calibration_end]
    test_ids = test_ids[~np.isin(test_ids, calibration_ids)]
    return calibration_ids, test_ids


def build_window_statistical_features(windows: np.ndarray) -> np.ndarray:
    """
    Вычисляет статистические признаки для каждого окна многоканального временного ряда.
    
    Для каждого окна (n_windows, window_size, n_channels) вычисляет по каждому каналу:
    - среднее,
    - стандартное отклонение,
    - минимум,
    - максимум,
    - разность между последним и первым отсчётами (наклон/размах).
    
    Результат конкатенируется по оси каналов, т.е. для каждого окна возвращается вектор
    размерности 5 * n_channels.
    
    Параметры:
        windows : np.ndarray
            Массив формы (n_windows, window_size, n_channels).
    
    Возвращает:
        np.ndarray
            Массив формы (n_windows, 5 * n_channels).
    
    Исключения:
        ValueError
            Если входной массив не трёхмерный.
    """
    array = np.asarray(windows, dtype=float)
    if array.ndim != 3:
        raise ValueError('Detection windows must have shape [n_windows, window_size, n_channels].')
    mean = np.mean(array, axis=1)
    std = np.std(array, axis=1)
    minimum = np.min(array, axis=1)
    maximum = np.max(array, axis=1)
    slope = array[:, -1, :] - array[:, 0, :]
    return np.concatenate((mean, std, minimum, maximum, slope), axis=1)


def infer_regime_segments(
        values: Sequence[float] | np.ndarray,
        *,
        volatility_window: int = 16,
        transition_quantile: float = 0.85,
) -> tuple[RegimeSegment, ...]:
    """
    Выделяет сегменты временного ряда, характеризующие режимы: высокий уровень (high_load),
    низкий уровень (low_load), переходный (transition) или стабильный (stable).

    Алгоритм:
    1. Берёт среднее по каналам (если многоканальный).
    2. Вычисляет скользящую волатильность (rolling std) с окном volatility_window.
    3. Вычисляет абсолютный наклон ряда abs(diff()).
    4. Порог перехода — quantile абсолютного наклона (transition_quantile).
    5. Квантили 75% и 25% среднего по каналам сигнала задают границы high/low.
    6. Размечает точки: transition, иначе stable, потом stable с высоким/низким уровнем.
    7. Склеивает соседние точки с одинаковой меткой в сегменты.

    Параметры:
        values : Sequence[float] или np.ndarray
            Входной ряд (1D или 2D или ND) - ensure_detection_array сожмёт до 2D.
        volatility_window : int, default=16
            Размер окна для расчёта волатильности. Не меньше 3.
        transition_quantile : float, default=0.85
            Квантиль абсолютного наклона для определения переходов (0..1).

    Возвращает:
        tuple[RegimeSegment, ...]
            Кортеж объектов RegimeSegment с индексами, меткой, средним уровнем,
            волатильностью и наклоном внутри сегмента.
    """
    series = ensure_detection_array(values)
    regime_signal = np.mean(series, axis=1)
    slope = np.diff(regime_signal, prepend=regime_signal[0])
    volatility = (
        pd.Series(regime_signal)
        .rolling(window=max(3, int(volatility_window)), min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    abs_slope = np.abs(slope)
    transition_threshold = float(np.quantile(abs_slope, transition_quantile)) if len(abs_slope) else 0.0
    high_level = float(np.quantile(regime_signal, 0.75))
    low_level = float(np.quantile(regime_signal, 0.25))

    labels = np.full(series.shape[0], 'stable', dtype=object)
    labels[abs_slope >= transition_threshold] = 'transition'
    labels[(labels == 'stable') & (regime_signal >= high_level)] = 'high_load'
    labels[(labels == 'stable') & (regime_signal <= low_level)] = 'low_load'

    segments: list[RegimeSegment] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segment_slice = slice(start, index)
            segment_values = regime_signal[segment_slice]
            segment_slope = slope[segment_slice]
            segment_volatility = volatility[segment_slice]
            segments.append(
                RegimeSegment(
                    start_index=int(start),
                    end_index=int(index - 1),
                    regime_label=str(labels[start]),
                    mean_level=float(np.mean(segment_values)),
                    volatility=float(np.mean(segment_volatility)),
                    slope=float(np.mean(segment_slope)),
                )
            )
            start = index
    return tuple(segments)


def align_window_scores_to_points(
        window_scores: Sequence[float] | np.ndarray,
        batch: DetectionWindowBatch,
) -> np.ndarray:
    """
    Преобразует оценки (аномалий), вычисленные для каждого окна, в поточечный ряд.

    Для каждой точки временного ряда усредняет оценки всех окон, которые покрывают эту точку.
    Точки без покрытия получают значение 0 (деление на 1 предотвращает разрыв).

    Параметры:
        window_scores : Sequence[float] или np.ndarray
            Оценки для каждого окна (длина равна batch.n_windows).
        batch : DetectionWindowBatch
            Пакет окон, содержащий window_indices и original_length.

    Возвращает:
        np.ndarray
            Массив длины original_length, где каждому индексу сопоставлена средняя оценка.

    Исключения:
        ValueError
            Если длина window_scores не совпадает с batch.n_windows.
    """
    scores = np.asarray(window_scores, dtype=float).reshape(-1)
    if scores.shape[0] != batch.n_windows:
        raise ValueError('The number of window scores must match the number of detection windows.')
    point_scores = np.zeros(batch.original_length, dtype=float)
    point_counts = np.zeros(batch.original_length, dtype=float)
    for score, (start, end) in zip(scores, batch.window_indices):
        point_scores[start:end] += float(score)
        point_counts[start:end] += 1.0
    point_counts = np.where(point_counts == 0.0, 1.0, point_counts)
    return point_scores / point_counts


def estimate_detection_threshold(
        scores: Sequence[float] | np.ndarray,
        *,
        strategy: str = 'mad',
        quantile: float = 0.99,
        regime_labels: Sequence[str] | None = None,
) -> float:
    """
    Вычисляет порог для бинаризации аномалий на основе оценок.

    Поддерживаемые стратегии:
    - 'mad' : медиана + 3 * MAD (медианное абсолютное отклонение); если MAD близок к нулю,
               использует стандартное отклонение.
    - 'quantile' : заданный квантиль распределения оценок.
    - 'regime_conditional' : применяет стратегию 'mad' только к точкам, не отмеченным как 'transition'
                             (если переданы regime_labels).
    - 'domain_calibrated' : среднее + 2.5 * стандартное отклонение.

    Параметры:
        scores : Sequence[float] или np.ndarray
            Вектор оценок (поточечных) длины original_length.
        strategy : str, default='mad'
            Название стратегии.
        quantile : float, default=0.99
            Квантиль (используется для strategy='quantile').
        regime_labels : Sequence[str] или None
            Метки режимов для каждой точки (требуется для 'regime_conditional').

    Возвращает:
        float
            Вычисленный порог.

    Исключения:
        ValueError
            Если стратегия не поддерживается.
    """
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    normalized_strategy = str(strategy).lower()
    if normalized_strategy == 'mad':
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median + 3.0 * (mad if mad > 1e-8 else np.std(values))
    if normalized_strategy == 'quantile':
        return float(np.quantile(values, quantile))
    if normalized_strategy == 'regime_conditional':
        if regime_labels is None:
            return estimate_detection_threshold(values, strategy='mad')
        labels = np.asarray(regime_labels, dtype=object).reshape(-1)
        if labels.shape[0] != values.shape[0]:
            return estimate_detection_threshold(values, strategy='mad')
        stable_scores = values[labels != 'transition']
        if stable_scores.size == 0:
            stable_scores = values
        return estimate_detection_threshold(stable_scores, strategy='mad')
    if normalized_strategy == 'domain_calibrated':
        return float(np.mean(values) + 2.5 * np.std(values))
    raise ValueError(f'Unsupported detection calibration strategy: {strategy}')


def build_anomaly_score_series(
        point_scores: Sequence[float] | np.ndarray,
        *,
        threshold: float,
        calibration_strategy: str,
        metadata: dict[str, Any] | None = None,
) -> AnomalyScoreSeries:
    """
    Формирует объект AnomalyScoreSeries из поточечных оценок и порога.

    Сравнивает каждую оценку с порогом и создаёт бинарную метку (1 если score >= threshold, иначе 0).

    Параметры:
        point_scores : Sequence[float] или np.ndarray
            Оценки для каждой точки временного ряда.
        threshold : float
            Порог детекции.
        calibration_strategy : str
            Название стратегии, использованной для получения порога (сохраняется в метаданных).
        metadata : dict или None
            Дополнительные метаданные.

    Возвращает:
        AnomalyScoreSeries
            Объект, содержащий кортежи оценок и меток, порог, стратегию и метаданные.
    """
    scores = np.asarray(point_scores, dtype=float).reshape(-1)
    labels = (scores >= float(threshold)).astype(int)
    return AnomalyScoreSeries(
        scores=tuple(float(value) for value in scores.tolist()),
        labels=tuple(int(value) for value in labels.tolist()),
        threshold=float(threshold),
        calibration_strategy=str(calibration_strategy),
        metadata=dict(metadata or {}),
    )


def detect_events_from_score_series(
        score_series: AnomalyScoreSeries,
        *,
        min_event_length: int = 1,
        regime_segments: Sequence[RegimeSegment] = (),
) -> tuple[DetectionEvent, ...]:
    """
    Обнаруживает непрерывные аномальные участки (события) на основе бинарной метки.

    Событием считается последовательность точек с меткой 1, длина которой не меньше min_event_length.
    Для каждого события определяются:
    - начальный и конечный индексы,
    - пиковый индекс (максимальная оценка внутри события),
    - пиковое и среднее значение оценки,
    - метка режима, соответствующая позиции пика (если переданы regime_segments).

    Параметры:
        score_series : AnomalyScoreSeries
            Серия с оценками и метками.
        min_event_length : int, default=1
            Минимальная длина события (количество последовательных аномальных точек).
        regime_segments : Sequence[RegimeSegment], default=()
            Сегменты режимов для определения regime_label события.

    Возвращает:
        tuple[DetectionEvent, ...]
            Кортеж обнаруженных событий, отсортированных по времени.
    """
    labels = np.asarray(score_series.labels, dtype=int).reshape(-1)
    scores = np.asarray(score_series.scores, dtype=float).reshape(-1)
    events: list[DetectionEvent] = []
    start = None
    for index, label in enumerate(labels):
        if label == 1 and start is None:
            start = index
        elif label == 0 and start is not None:
            if index - start >= int(min_event_length):
                events.append(_build_detection_event(start, index - 1, scores, regime_segments))
            start = None
    if start is not None and len(labels) - start >= int(min_event_length):
        events.append(_build_detection_event(start, len(labels) - 1, scores, regime_segments))
    return tuple(events)


def _build_detection_event(
        start: int,
        end: int,
        scores: np.ndarray,
        regime_segments: Sequence[RegimeSegment],
) -> DetectionEvent:
    """
    Вспомогательная функция: создаёт один объект DetectionEvent для заданного интервала.

    Вычисляет пик (максимум scores), среднее значение и определяет метку режима
    по пересечению с regime_segments.

    Параметры:
        start : int
            Начальный индекс события.
        end : int
            Конечный индекс события (включительно).
        scores : np.ndarray
            Массив оценок для всего ряда.
        regime_segments : Sequence[RegimeSegment]
            Сегменты режимов.

    Возвращает:
        DetectionEvent
            Сформированный объект события.
    """
    segment_scores = scores[start:end + 1]
    peak_offset = int(np.argmax(segment_scores))
    peak_index = int(start + peak_offset)
    regime_label = None
    for segment in regime_segments:
        if segment.start_index <= peak_index <= segment.end_index:
            regime_label = segment.regime_label
            break
    return DetectionEvent(
        start_index=int(start),
        end_index=int(end),
        peak_index=peak_index,
        peak_score=float(np.max(segment_scores)),
        mean_score=float(np.mean(segment_scores)),
        regime_label=regime_label,
    )


def domain_invariant_scale(
        values: Sequence[float] | np.ndarray,
        *,
        reference_values: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Масштабирует данные, делая их инвариантными к сдвигу и масштабу относительно опорного распределения.

    Используется преобразование: (x - median) / MAD, где MAD — медианное абсолютное отклонение.
    Если reference_values не задан, используется сама серия.
    MAD, близкие к нулю, заменяются на 1.0.

    Параметры:
        values : Sequence[float] или np.ndarray
            Данные для масштабирования (1D или 2D).
        reference_values : Sequence[float] или np.ndarray или None
            Опорные данные для оценки медианы и MAD. Если None — используются values.

    Возвращает:
        np.ndarray
            Масштабированные данные той же формы, что и values.
    """
    series = ensure_detection_array(values)
    reference = ensure_detection_array(reference_values) if reference_values is not None else series
    median = np.median(reference, axis=0)
    mad = np.median(np.abs(reference - median), axis=0)
    mad = np.where(mad < 1e-8, 1.0, mad)
    return (series - median) / mad


def coral_feature_align(
        source_features: Sequence[float] | np.ndarray,
        target_features: Sequence[float] | np.ndarray,
        *,
        epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Выполняет выравнивание признаков методом CORAL (CORrelation ALignment).
    
    Приводит ковариацию и среднее исходных признаков к ковариации и среднему целевых,
    используя линейное преобразование:
    1. Центрирует source и target.
    2. Вычисляет ковариационные матрицы с регуляризацией epsilon.
    3. Строит отбеливающее преобразование для source и окрашивающее для target.
    4. Применяет преобразование к центрированному source и добавляет среднее target.
    
    Параметры:
        source_features : Sequence[float] или np.ndarray
            Признаки исходного домена (shape n_source x d).
        target_features : Sequence[float] или np.ndarray
            Признаки целевого домена (shape n_target x d).
        epsilon : float, default=1e-6
            Регуляризация для ковариационных матриц (добавляется к диагонали).
    
    Возвращает:
        np.ndarray
            Выровненные признаки исходного домена (n_source x d).
    
    Исключения:
        ValueError
            Если входные массивы не двумерные.
    """
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError('CORAL feature alignment expects 2D feature matrices.')
    source_centered = source - np.mean(source, axis=0, keepdims=True)
    target_centered = target - np.mean(target, axis=0, keepdims=True)
    source_cov = np.cov(source_centered, rowvar=False) + epsilon * np.eye(source.shape[1])
    target_cov = np.cov(target_centered, rowvar=False) + epsilon * np.eye(target.shape[1])
    source_values, source_vectors = np.linalg.eigh(source_cov)
    target_values, target_vectors = np.linalg.eigh(target_cov)
    source_whitener = source_vectors @ np.diag(1.0 / np.sqrt(np.clip(source_values, epsilon, None))) @ source_vectors.T
    target_colorer = target_vectors @ np.diag(np.sqrt(np.clip(target_values, epsilon, None))) @ target_vectors.T
    aligned = source_centered @ source_whitener @ target_colorer + np.mean(target, axis=0, keepdims=True)
    return np.asarray(aligned, dtype=float)


def build_transfer_alignment_report(
        source_values: Sequence[float] | np.ndarray,
        target_values: Sequence[float] | np.ndarray,
        *,
        strategy: str = 'domain_invariant_scaling',
        source_domain: str = 'source',
        target_domain: str = 'target',
) -> TransferAlignmentReport:
    """
    Создаёт отчёт о различиях между исходным и целевым доменом для диагностики переноса.

    Вычисляет средние значения по каналам, количество наблюдений, сдвиг средних,
    а также включает в метаданные стандартные отклонения.

    Параметры:
        source_values : Sequence[float] или np.ndarray
            Данные исходного домена.
        target_values : Sequence[float] или np.ndarray
            Данные целевого домена.
        strategy : str, default='domain_invariant_scaling'
            Название стратегии выравнивания (сохраняется в отчёте).
        source_domain : str, default='source'
            Метка для исходного домена.
        target_domain : str, default='target'
            Метка для целевого домена.

    Возвращает:
        TransferAlignmentReport
            Объект отчёта с полями: strategy, source_domain, target_domain,
            n_source, n_target, source_channel_mean, target_channel_mean,
            mean_shift, metadata.
    """
    source = ensure_detection_array(source_values)
    target = ensure_detection_array(target_values)
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    return TransferAlignmentReport(
        strategy=str(strategy),
        source_domain=source_domain,
        target_domain=target_domain,
        n_source=int(source.shape[0]),
        n_target=int(target.shape[0]),
        source_channel_mean=tuple(float(value) for value in source_mean.tolist()),
        target_channel_mean=tuple(float(value) for value in target_mean.tolist()),
        mean_shift=tuple(float(value) for value in (target_mean - source_mean).tolist()),
        metadata={
            'source_channel_std': [float(value) for value in np.std(source, axis=0).tolist()],
            'target_channel_std': [float(value) for value in np.std(target, axis=0).tolist()],
        },
    )


def build_risk_feature_frame(
        *,
        events: Sequence[DetectionEvent],
        regime_segments: Sequence[RegimeSegment],
        score_series: AnomalyScoreSeries | None = None,
        node_name: str | None = None,
        domain_name: str | None = None,
) -> RiskFeatureFrame:
    """
    Формирует таблицу (RiskFeatureFrame) с признаками риска для каждого обнаруженного события.

    Для каждого события извлекаются:
    - индексы начала, конца, пика,
    - пиковая и средняя оценки,
    - длина события,
    - метка режима, средний уровень и волатильность режима (из regime_segments),
    - название узла (node_name) и домена (domain_name),
    - если передан score_series — порог и стратегия калибровки.

    Args:
        events : Sequence[DetectionEvent]
            Обнаруженные события.
        regime_segments : Sequence[RegimeSegment]
            Сегменты режимов для привязки regime_label и характеристик.
        score_series : AnomalyScoreSeries или None
            Серия оценок (для добавления threshold и calibration_strategy).
        node_name : str или None
            Имя узла (например, идентификатор оборудования).
        domain_name : str или None
            Имя домена.

    Returns:
        RiskFeatureFrame
            Объект с полями columns (кортеж названий столбцов),
            rows (кортеж словарей) и metadata (словарь с n_events и n_regime_segments).
    """
    rows: list[dict[str, Any]] = []
    regime_lookup = {
        (segment.start_index, segment.end_index): segment
        for segment in regime_segments
    }
    for event in events:
        matched_regime = next(
            (
                segment for key, segment in regime_lookup.items()
                if key[0] <= event.peak_index <= key[1]
            ),
            None,
        )
        row = {
            'event_start_index': int(event.start_index),
            'event_end_index': int(event.end_index),
            'event_peak_index': int(event.peak_index),
            'event_peak_score': float(event.peak_score),
            'event_mean_score': float(event.mean_score),
            'event_length': int(event.end_index - event.start_index + 1),
            'regime_label': event.regime_label or (matched_regime.regime_label if matched_regime else 'unknown'),
            'regime_mean_level': None if matched_regime is None else float(matched_regime.mean_level),
            'regime_volatility': None if matched_regime is None else float(matched_regime.volatility),
            'node_name': node_name or '',
            'domain_name': domain_name or '',
        }
        if score_series is not None:
            row['event_threshold'] = float(score_series.threshold)
            row['calibration_strategy'] = score_series.calibration_strategy
        rows.append(row)
    columns = tuple(rows[0].keys()) if rows else (
        'event_start_index',
        'event_end_index',
        'event_peak_index',
        'event_peak_score',
        'event_mean_score',
        'event_length',
        'regime_label',
        'regime_mean_level',
        'regime_volatility',
        'node_name',
        'domain_name',
    )
    return RiskFeatureFrame(
        columns=columns,
        rows=tuple(rows),
        metadata={
            'n_events': len(rows),
            'n_regime_segments': len(regime_segments),
        },
    )

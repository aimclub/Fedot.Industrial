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
            raise ValueError(
                'FEDOT InputData is unavailable in the current environment.')
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
            metadata={'idx': getattr(
                input_data, 'idx', None), **dict(metadata or {})},
        )


def ensure_detection_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
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
    if series_length < 4:
        return max(1, int(series_length))
    if window_size is not None:
        return max(2, min(int(window_size), int(series_length)))
    if window_size_percent is not None:
        resolved = int(
            round(series_length * float(window_size_percent) / 100.0))
        return max(2, min(resolved, int(series_length)))
    default_window = max(minimum_window_size, int(round(series_length * 0.1)))
    return min(default_window, int(series_length))


def resolve_detection_stride(window_size: int, stride: int | None = None) -> int:
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
        channel_names=tuple(channel_names or tuple(
            f'channel_{index}' for index in range(series.shape[1]))),
        metadata=dict(metadata or {}),
    )


def split_detection_batch(
        batch: DetectionWindowBatch,
        split_spec: DetectionSplitSpec,
) -> tuple[DetectionWindowBatch, DetectionWindowBatch, DetectionWindowBatch | None]:
    n_windows = batch.n_windows
    if n_windows <= 0:
        raise ValueError('Detection split requires at least one window.')

    def _metadata_for_indices(indices: np.ndarray, split_name: str) -> dict[str, Any]:
        metadata = dict(batch.metadata)
        for key, value in list(metadata.items()):
            if isinstance(value, np.ndarray) and value.shape[:1] == (n_windows,):
                metadata[key] = value[indices].tolist()
            elif isinstance(value, (list, tuple)) and len(value) == n_windows:
                metadata[key] = [value[int(index)] for index in indices]
        metadata.update({
            'split_kind': split_spec.kind.value,
            'split_name': split_name,
            'selected_window_ids': [int(index) for index in indices.tolist()],
        })
        return metadata

    def _take(indices: Sequence[int], split_name: str) -> DetectionWindowBatch | None:
        selected = np.asarray(list(indices), dtype=int)
        if selected.size == 0:
            return None
        selected_mask = batch.mask
        if isinstance(batch.mask, np.ndarray) and batch.mask.shape[:1] == (n_windows,):
            selected_mask = batch.mask[selected]
        return DetectionWindowBatch(
            windows=batch.windows[selected],
            window_indices=batch.window_indices[selected],
            original_length=batch.original_length,
            window_size=batch.window_size,
            stride=batch.stride,
            channel_names=batch.channel_names,
            mask=selected_mask,
            metadata=_metadata_for_indices(selected, split_name),
        )

    if split_spec.kind is DetectionSplitKind.DOMAIN_HOLDOUT:
        domains = batch.metadata.get('window_domains')
        if domains is None:
            raise ValueError(
                'Domain holdout split requires window_domains metadata.')
        if len(domains) != n_windows:
            raise ValueError(
                'window_domains metadata must match the number of detection windows.')
        target_domain = split_spec.target_domain or str(domains[-1])
        train_ids = [index for index, domain in enumerate(
            domains) if str(domain) != str(target_domain)]
        target_ids = [index for index, domain in enumerate(
            domains) if str(domain) == str(target_domain)]
        n_calibration = max(1, int(
            round(len(target_ids) * split_spec.calibration_fraction))) if target_ids else 0
        n_calibration = min(len(target_ids), n_calibration)
        train_batch = _take(train_ids, 'train')
        calibration_batch = _take(target_ids[:n_calibration], 'calibration')
        test_batch = _take(target_ids[n_calibration:], 'test')
        if train_batch is None:
            raise ValueError(
                'Detection domain holdout produced an empty train batch.')
        return train_batch, calibration_batch or train_batch, test_batch

    if split_spec.kind is DetectionSplitKind.TEMPORAL and split_spec.prevent_future_leakage:
        train_boundary = int(
            round(batch.original_length * split_spec.train_fraction))
        calibration_boundary = int(round(
            batch.original_length *
            min(1.0, split_spec.train_fraction +
                split_spec.calibration_fraction)
        ))
        train_ids = np.flatnonzero(
            batch.window_indices[:, 1] <= train_boundary)
        calibration_ids = np.flatnonzero(
            (batch.window_indices[:, 0] >= train_boundary)
            & (batch.window_indices[:, 1] <= calibration_boundary)
        )
        test_ids = np.flatnonzero(
            batch.window_indices[:, 0] >= calibration_boundary)
        train_batch = _take(train_ids, 'train')
        calibration_batch = _take(calibration_ids, 'calibration')
        test_batch = _take(test_ids, 'test')
        if train_batch is None:
            raise ValueError(
                'Detection temporal split produced an empty train batch.')
        return train_batch, calibration_batch or train_batch, test_batch

    n_train = max(1, int(round(n_windows * split_spec.train_fraction)))
    remaining = max(0, n_windows - n_train)
    n_calibration = int(round(remaining * split_spec.calibration_fraction /
                        max(1e-8, 1.0 - split_spec.train_fraction)))
    n_calibration = min(remaining, max(0, n_calibration))
    if split_spec.kind is DetectionSplitKind.HOLDOUT:
        ordered_ids = np.random.default_rng(
            split_spec.random_seed).permutation(n_windows)
    else:
        ordered_ids = np.arange(n_windows, dtype=int)

    train_batch = _take(sorted(ordered_ids[:n_train].tolist()), 'train')
    calibration_batch = _take(
        sorted(ordered_ids[n_train:n_train + n_calibration].tolist()), 'calibration')
    test_batch = _take(
        sorted(ordered_ids[n_train + n_calibration:].tolist()), 'test')
    if train_batch is None:
        raise ValueError('Detection split produced an empty train batch.')
    return train_batch, calibration_batch or train_batch, test_batch


def build_window_statistical_features(windows: np.ndarray) -> np.ndarray:
    array = np.asarray(windows, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            'Detection windows must have shape [n_windows, window_size, n_channels].')
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
    transition_threshold = float(np.quantile(
        abs_slope, transition_quantile)) if len(abs_slope) else 0.0
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
    scores = np.asarray(window_scores, dtype=float).reshape(-1)
    if scores.shape[0] != batch.n_windows:
        raise ValueError(
            'The number of window scores must match the number of detection windows.')
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
    labels = np.asarray(score_series.labels, dtype=int).reshape(-1)
    scores = np.asarray(score_series.scores, dtype=float).reshape(-1)
    events: list[DetectionEvent] = []
    start = None
    for index, label in enumerate(labels):
        if label == 1 and start is None:
            start = index
        elif label == 0 and start is not None:
            if index - start >= int(min_event_length):
                events.append(_build_detection_event(
                    start, index - 1, scores, regime_segments))
            start = None
    if start is not None and len(labels) - start >= int(min_event_length):
        events.append(_build_detection_event(
            start, len(labels) - 1, scores, regime_segments))
    return tuple(events)


def _build_detection_event(
        start: int,
        end: int,
        scores: np.ndarray,
        regime_segments: Sequence[RegimeSegment],
) -> DetectionEvent:
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
    series = ensure_detection_array(values)
    reference = ensure_detection_array(
        reference_values) if reference_values is not None else series
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
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(
            'CORAL feature alignment expects 2D feature matrices.')
    source_centered = source - np.mean(source, axis=0, keepdims=True)
    target_centered = target - np.mean(target, axis=0, keepdims=True)
    source_cov = np.cov(source_centered, rowvar=False) + \
        epsilon * np.eye(source.shape[1])
    target_cov = np.cov(target_centered, rowvar=False) + \
        epsilon * np.eye(target.shape[1])
    source_values, source_vectors = np.linalg.eigh(source_cov)
    target_values, target_vectors = np.linalg.eigh(target_cov)
    source_whitener = source_vectors @ np.diag(1.0 / np.sqrt(
        np.clip(source_values, epsilon, None))) @ source_vectors.T
    target_colorer = target_vectors @ np.diag(
        np.sqrt(np.clip(target_values, epsilon, None))) @ target_vectors.T
    aligned = source_centered @ source_whitener @ target_colorer + \
        np.mean(target, axis=0, keepdims=True)
    return np.asarray(aligned, dtype=float)


def build_transfer_alignment_report(
        source_values: Sequence[float] | np.ndarray,
        target_values: Sequence[float] | np.ndarray,
        *,
        strategy: str = 'domain_invariant_scaling',
        source_domain: str = 'source',
        target_domain: str = 'target',
) -> TransferAlignmentReport:
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
        source_channel_mean=tuple(float(value)
                                  for value in source_mean.tolist()),
        target_channel_mean=tuple(float(value)
                                  for value in target_mean.tolist()),
        mean_shift=tuple(float(value)
                         for value in (target_mean - source_mean).tolist()),
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

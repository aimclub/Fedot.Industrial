from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.svm import OneClassSVM

from fedot_ind.core.models.detection.runtime import (
    AnomalyScoreSeries,
    DetectionEvent,
    RiskFeatureFrame,
    align_window_scores_to_points,
    build_anomaly_score_series,
    build_detection_window_batch,
    build_risk_feature_frame,
    build_transfer_alignment_report,
    detect_events_from_score_series,
    domain_invariant_scale,
    ensure_detection_array,
    estimate_detection_threshold,
    infer_regime_segments,
    resolve_detection_stride,
    resolve_detection_window_size,
)
from fedot_ind.core.models.detection.stage_tuning import build_detection_stage_tuning_plan
from fedot_ind.core.repository.detection_registry import detection_family_for

try:  # pragma: no cover - optional in lightweight environments
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = TensorDataset = None


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _regime_labels_from_segments(segments, length: int) -> np.ndarray:
    labels = np.full(int(length), 'stable', dtype=object)
    for segment in segments:
        labels[segment.start_index:segment.end_index + 1] = segment.regime_label
    return labels


def _operation_params_to_dict(params) -> dict[str, Any]:
    """Convert FEDOT OperationParameters or plain mappings into a safe dict."""
    if params is None:
        return {}
    if hasattr(params, 'to_dict'):
        return dict(params.to_dict())
    return dict(params)


class BaseRuntimeAnomalyDetector(ModelImplementation, ABC):
    canonical_name: str = 'runtime_detection_model'
    default_representation_mode: str = 'statistical'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.params = params or OperationParameters()
        self.window_size = self.params.get('window_size')
        self.window_size_percent = self.params.get('window_size_percent', self.params.get('window_length', 10))
        self.stride = self.params.get('stride')
        self.calibration_strategy = self.params.get('calibration_strategy', 'mad')
        self.threshold_quantile = self.params.get('threshold_quantile', 0.99)
        self.min_event_length = int(self.params.get('min_event_length', 1))
        self.transfer_strategy = self.params.get('transfer_strategy', 'domain_invariant_scaling')
        self.representation_mode = self.params.get('representation_mode', self.default_representation_mode)
        self.enable_regime_segmentation = bool(self.params.get('enable_regime_segmentation', True))

    @property
    def family(self) -> str:
        return detection_family_for(self.canonical_name)


    def fit(self, input_data: np.ndarray) -> None:
        series = self._prepare_series(input_data, fit_stage=True)
        batch = self._build_batch(series, metadata={'fit_stage': True})
        self.training_batch_ = batch
        self.regime_segments_ = infer_regime_segments(series) if self.enable_regime_segmentation else ()
        features, representation_diagnostics = self._build_representation(batch, fit_stage=True)
        self._fit_scoring_model(features, batch=batch)
        window_scores = self._score_windows(features, batch=batch)
        point_scores = align_window_scores_to_points(window_scores, batch)
        regime_labels = _regime_labels_from_segments(self.regime_segments_, len(point_scores))
        self.threshold_ = estimate_detection_threshold(
            point_scores,
            strategy=self.calibration_strategy,
            quantile=float(self.threshold_quantile),
            regime_labels=regime_labels,
        )
        self.training_score_series_ = build_anomaly_score_series(
            point_scores,
            threshold=self.threshold_,
            calibration_strategy=self.calibration_strategy,
            metadata={'fit_stage': True},
        )
        self.training_events_ = detect_events_from_score_series(
            self.training_score_series_,
            min_event_length=self.min_event_length,
            regime_segments=self.regime_segments_,
        )
        self.risk_feature_frame_ = build_risk_feature_frame(
            events=self.training_events_,
            regime_segments=self.regime_segments_,
            score_series=self.training_score_series_,
            node_name=self.params.get('node_name'),
            domain_name=self.params.get('domain_name'),
        )
        self.transfer_report_ = build_transfer_alignment_report(
            series,
            self.scaling_reference_,
            strategy=self.transfer_strategy,
            source_domain='fit',
            target_domain='reference',
        )
        self.stage_diagnostics_ = {
            'data_quality': {
                'n_samples': int(series.shape[0]),
                'n_channels': int(series.shape[1]),
                'window_size': int(batch.window_size),
                'stride': int(batch.stride),
            },
            'regime_segmentation': {
                'n_segments': len(self.regime_segments_),
                'segment_labels': [segment.regime_label for segment in self.regime_segments_],
            },
            'representation': representation_diagnostics,
            'anomaly_scoring': self._score_diagnostics(),
            'calibration': {
                'strategy': self.calibration_strategy,
                'threshold': float(self.threshold_),
                'threshold_quantile': float(self.threshold_quantile),
            },
            'event_aggregation': {
                'n_events': len(self.training_events_),
                'min_event_length': int(self.min_event_length),
            },
            'transfer_alignment': self.transfer_report_.to_dict(),
            'interpretation': {
                'risk_feature_columns': list(self.risk_feature_frame_.columns),
                'risk_feature_rows': len(self.risk_feature_frame_.rows),
            },
        }

    def predict(self, values: np.ndarray | list[float]) -> np.ndarray:
        score_series = self.score_series_on_values(values)
        labels = np.asarray(score_series.labels, dtype=int).reshape(-1, 1)
        return labels

    def predict_for_fit(self, values: np.ndarray | list[float]):
        return self.score_samples(values)

    def predict_proba(self, values: np.ndarray | list[float]) -> np.ndarray:
        return self.score_samples(values)

    def score_samples(self, values: np.ndarray | list[float]) -> np.ndarray:
        score_series = self.score_series_on_values(values)
        scores = np.asarray(score_series.scores, dtype=float)
        reference_scale = np.std(scores) if np.std(scores) > 1e-8 else max(float(score_series.threshold), 1.0)
        anomaly_probability = _sigmoid((scores - float(score_series.threshold)) / reference_scale)
        return np.column_stack((1.0 - anomaly_probability, anomaly_probability))

    def score_series_on_values(self, values: np.ndarray | list[float]) -> AnomalyScoreSeries:
        series = self._prepare_series(values, fit_stage=False)
        batch = self._build_batch(series, metadata={'fit_stage': False})
        features, _ = self._build_representation(batch, fit_stage=False)
        window_scores = self._score_windows(features, batch=batch)
        point_scores = align_window_scores_to_points(window_scores, batch)
        return build_anomaly_score_series(
            point_scores,
            threshold=float(self.threshold_),
            calibration_strategy=self.calibration_strategy,
            metadata={'fit_stage': False},
        )

    def detect_events_on_values(self, values: np.ndarray | list[float]) -> tuple[DetectionEvent, ...]:
        score_series = self.score_series_on_values(values)
        segments = infer_regime_segments(values) if self.enable_regime_segmentation else ()
        return detect_events_from_score_series(
            score_series,
            min_event_length=self.min_event_length,
            regime_segments=segments,
        )

    def get_stage_tuning_plan(self) -> dict[str, Any]:
        return build_detection_stage_tuning_plan(self.canonical_name, _operation_params_to_dict(self.params)).to_dict()

    def get_risk_feature_frame(self) -> RiskFeatureFrame:
        return self.risk_feature_frame_

    def get_stage_diagnostics(self) -> dict[str, Any]:
        return dict(self.stage_diagnostics_)

    def _build_batch(self, values: np.ndarray, metadata: dict[str, Any]) -> Any:
        window_size = resolve_detection_window_size(
            values.shape[0],
            window_size=self.window_size,
            window_size_percent=self.window_size_percent,
        )
        stride = resolve_detection_stride(window_size, self.stride)
        return build_detection_window_batch(values, window_size=window_size, stride=stride, metadata=metadata)

    def _prepare_series(self, values: np.ndarray | list[float], *, fit_stage: bool) -> np.ndarray:
        series = ensure_detection_array(values)
        if fit_stage or not hasattr(self, 'scaling_reference_'):
            self.scaling_reference_ = np.asarray(series, dtype=float)
        if self.transfer_strategy == 'domain_invariant_scaling':
            return domain_invariant_scale(series, reference_values=self.scaling_reference_)
        return np.asarray(series, dtype=float)

    def _build_representation(self, batch, *, fit_stage: bool) -> tuple[np.ndarray, dict[str, Any]]:
        del fit_stage
        if self.representation_mode == 'flatten':
            features = batch.flattened_features
        elif self.representation_mode == 'identity':
            features = batch.windows
        else:
            features = batch.statistical_features
        return np.asarray(features, dtype=float), {
            'representation_mode': self.representation_mode,
            'feature_shape': tuple(int(value) for value in np.asarray(features).shape),
        }

    @abstractmethod
    def _fit_scoring_model(self, features: np.ndarray, *, batch) -> None:
        raise NotImplementedError

    @abstractmethod
    def _score_windows(self, features: np.ndarray, *, batch) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _score_diagnostics(self) -> dict[str, Any]:
        raise NotImplementedError


class FeatureIsolationForestDetector(BaseRuntimeAnomalyDetector):
    canonical_name = 'feature_iforest_detector'
    default_representation_mode = 'statistical'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_estimators = int(self.params.get('n_estimators', 200))
        self.contamination = self.params.get('contamination', 'auto')
        self.random_state = int(self.params.get('random_state', 42))
        self.n_jobs = int(self.params.get('n_jobs', -1))

    def _fit_scoring_model(self, features: np.ndarray, *, batch) -> None:
        del batch
        self.model_impl = SklearnIsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_impl.fit(features)

    def _score_windows(self, features: np.ndarray, *, batch) -> np.ndarray:
        del batch
        return -self.model_impl.score_samples(features)

    def _score_diagnostics(self) -> dict[str, Any]:
        return {
            'model_family': self.family,
            'model_name': self.canonical_name,
            'n_estimators': int(self.n_estimators),
            'contamination': self.contamination,
        }


class FeatureOneClassDetector(BaseRuntimeAnomalyDetector):
    canonical_name = 'feature_oneclass_detector'
    default_representation_mode = 'statistical'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.nu = float(self.params.get('nu', 0.05))
        self.kernel = str(self.params.get('kernel', 'rbf'))
        self.gamma = self.params.get('gamma', 'scale')

    def _fit_scoring_model(self, features: np.ndarray, *, batch) -> None:
        del batch
        self.model_impl = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
        )
        self.model_impl.fit(features)

    def _score_windows(self, features: np.ndarray, *, batch) -> np.ndarray:
        del batch
        return -self.model_impl.decision_function(features).reshape(-1)

    def _score_diagnostics(self) -> dict[str, Any]:
        return {
            'model_family': self.family,
            'model_name': self.canonical_name,
            'kernel': self.kernel,
            'nu': float(self.nu),
        }


@dataclass
class _TorchTrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    device: str


class _TorchAutoencoderDetector(BaseRuntimeAnomalyDetector, ABC):
    default_representation_mode = 'identity'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.epochs = int(self.params.get('epochs', 20))
        self.batch_size = int(self.params.get('batch_size', 32))
        self.learning_rate = float(self.params.get('learning_rate', 1e-3))
        self.latent_dim = int(self.params.get('latent_dim', 16))
        self.device = str(self.params.get('device', 'cpu'))

    def _build_representation(self, batch, *, fit_stage: bool) -> tuple[np.ndarray, dict[str, Any]]:
        del fit_stage
        return np.asarray(batch.windows, dtype=float), {
            'representation_mode': 'identity',
            'feature_shape': tuple(int(value) for value in batch.windows.shape),
        }

    def _fit_scoring_model(self, features: np.ndarray, *, batch) -> None:
        del batch
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:  # pragma: no cover
            raise ValueError('torch is required for neural anomaly detectors.')
        training_data = torch.from_numpy(np.asarray(features, dtype=np.float32).transpose(0, 2, 1))
        dataset = TensorDataset(training_data)
        loader = DataLoader(dataset, batch_size=min(len(dataset), self.batch_size), shuffle=True)
        self.model_impl = self._build_torch_model(training_data.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model_impl.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model_impl.train()
        for _ in range(self.epochs):
            for (batch_tensor,) in loader:
                batch_tensor = batch_tensor.to(self.device)
                optimizer.zero_grad()
                prediction = self.model_impl(batch_tensor)
                loss = loss_fn(prediction, batch_tensor)
                loss.backward()
                optimizer.step()

    def _score_windows(self, features: np.ndarray, *, batch) -> np.ndarray:
        del batch
        windows = torch.from_numpy(np.asarray(features, dtype=np.float32).transpose(0, 2, 1)).to(self.device)
        self.model_impl.eval()
        with torch.no_grad():
            reconstructed = self.model_impl(windows)
            residual = torch.mean((reconstructed - windows) ** 2, dim=(1, 2))
        return residual.detach().cpu().numpy().reshape(-1)

    def _score_diagnostics(self) -> dict[str, Any]:
        return {
            'model_family': self.family,
            'model_name': self.canonical_name,
            'epochs': int(self.epochs),
            'batch_size': int(self.batch_size),
            'learning_rate': float(self.learning_rate),
            'latent_dim': int(self.latent_dim),
        }

    @abstractmethod
    def _build_torch_model(self, n_channels: int):
        raise NotImplementedError


class _ConvWindowAutoencoder(nn.Module):  # pragma: no cover - exercised through detectors
    def __init__(self, n_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, values):
        return self.decoder(self.encoder(values))


class _TCNResidualBlock(nn.Module):  # pragma: no cover - exercised through detectors
    def __init__(self, n_channels: int, hidden_channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(n_channels, hidden_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, n_channels, kernel_size=3, padding=padding, dilation=dilation),
        )
        self.activation = nn.ReLU()

    def forward(self, values):
        return self.activation(self.block(values) + values)


class _TCNWindowAutoencoder(nn.Module):  # pragma: no cover - exercised through detectors
    def __init__(self, n_channels: int, latent_dim: int, kernel_size: int, num_filters: int, num_levels: int):
        super().__init__()
        del latent_dim, kernel_size
        blocks = [
            _TCNResidualBlock(n_channels, num_filters, dilation=2 ** level)
            for level in range(num_levels)
        ]
        self.network = nn.Sequential(
            *blocks,
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
        )

    def forward(self, values):
        return self.network(values)


class ConvAutoencoderDetector(_TorchAutoencoderDetector):
    canonical_name = 'conv_autoencoder_detector'

    def _build_torch_model(self, n_channels: int):
        return _ConvWindowAutoencoder(n_channels=n_channels, latent_dim=self.latent_dim)


class TCNAutoencoderDetector(_TorchAutoencoderDetector):
    canonical_name = 'tcn_autoencoder_detector'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.kernel_size = int(self.params.get('kernel_size', 3))
        self.num_filters = int(self.params.get('num_filters', 32))
        self.num_levels = int(self.params.get('num_levels', 3))

    def _build_torch_model(self, n_channels: int):
        return _TCNWindowAutoencoder(
            n_channels=n_channels,
            latent_dim=self.latent_dim,
            kernel_size=self.kernel_size,
            num_filters=self.num_filters,
            num_levels=self.num_levels,
        )

    def _score_diagnostics(self) -> dict[str, Any]:
        diagnostics = super()._score_diagnostics()
        diagnostics.update(
            {
                'kernel_size': int(self.kernel_size),
                'num_filters': int(self.num_filters),
                'num_levels': int(self.num_levels),
            }
        )
        return diagnostics


def build_detection_input_data(
        values: np.ndarray | list[float],
        *,
        target: np.ndarray | list[float] | None = None,
) -> InputData:
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    series = ensure_detection_array(values)
    return InputData(
        idx=np.arange(series.shape[0]),
        features=series,
        target=None if target is None else np.asarray(target),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

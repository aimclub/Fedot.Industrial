from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import torch

try:  # pragma: no cover - optional FEDOT runtime in lightweight envs
    from fedot.core.data.data import InputData, OutputData
    from fedot.core.repository.dataset_types import DataTypesEnum
except Exception:  # pragma: no cover
    InputData = OutputData = None


    class DataTypesEnum:  # type: ignore[override]
        table = 'table'
        ts = 'ts'


class ForecastingSplitKind(str, Enum):
    HOLDOUT = 'holdout'
    ROLLING_ORIGIN = 'rolling_origin'
    BLOCKED = 'blocked'


@dataclass(frozen=True)
class TensorDevicePolicy:
    device: str = 'cpu'
    dtype: str = 'float32'

    def resolve_device(self) -> torch.device:
        requested = str(self.device).lower()
        if requested.startswith('cuda') and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device('cpu')

    def resolve_dtype(self) -> torch.dtype:
        try:
            return getattr(torch, str(self.dtype))
        except AttributeError:  # pragma: no cover - defensive fallback
            return torch.float32


@dataclass(frozen=True)
class ForecastTensorBatch:
    history: torch.Tensor
    target: torch.Tensor | None
    forecast_horizon: int
    idx: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.history.device

    @property
    def channel_count(self) -> int:
        return int(self.history.shape[1])

    @property
    def series_length(self) -> int:
        return int(self.history.shape[0])

    def to(self, device_policy: TensorDevicePolicy) -> 'ForecastTensorBatch':
        device = device_policy.resolve_device()
        dtype = device_policy.resolve_dtype()
        return ForecastTensorBatch(
            history=self.history.to(device=device, dtype=dtype),
            target=None if self.target is None else self.target.to(device=device, dtype=dtype),
            forecast_horizon=int(self.forecast_horizon),
            idx=None if self.idx is None else self.idx.to(device=device),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'history_shape': tuple(int(value) for value in self.history.shape),
            'target_shape': None if self.target is None else tuple(int(value) for value in self.target.shape),
            'forecast_horizon': int(self.forecast_horizon),
            'device': str(self.device),
            'channel_count': self.channel_count,
            'series_length': self.series_length,
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class ForecastingOperationCapability:
    name: str
    stage: str
    tensor_native: bool = True
    supports_multivariate: bool = True
    supports_tuning: bool = True
    supports_cuda: bool = True


@dataclass(frozen=True)
class ForecastingSplitSpec:
    kind: ForecastingSplitKind = ForecastingSplitKind.HOLDOUT
    validation_horizon: int | None = None
    min_train_length: int | None = None


@dataclass(frozen=True)
class ForecastingEvaluationResult:
    metric_name: str
    metric_value: float
    per_horizon_metrics: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'metric_value': float(self.metric_value),
            'per_horizon_metrics': [float(value) for value in self.per_horizon_metrics],
            **self.metadata,
        }


@dataclass(frozen=True)
class TrajectoryTransformResult:
    features: torch.Tensor
    target: torch.Tensor | None
    latest_features: torch.Tensor
    window_size: int
    stride: int
    forecast_horizon: int
    channel_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'features_shape': tuple(int(value) for value in self.features.shape),
            'target_shape': None if self.target is None else tuple(int(value) for value in self.target.shape),
            'latest_features_shape': tuple(int(value) for value in self.latest_features.shape),
            'window_size': int(self.window_size),
            'stride': int(self.stride),
            'forecast_horizon': int(self.forecast_horizon),
            'channel_count': int(self.channel_count),
            **self.metadata,
        }


@dataclass(frozen=True)
class DecompositionResult:
    projected_features: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    strategy: str
    input_shape: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'projected_shape': tuple(int(value) for value in self.projected_features.shape),
            'basis_shape': tuple(int(value) for value in self.basis.shape),
            'input_shape': tuple(int(value) for value in self.input_shape),
            'strategy': str(self.strategy),
            **self.metadata,
        }


@dataclass(frozen=True)
class RankTruncationResult:
    projected_features: torch.Tensor
    basis: torch.Tensor
    reconstructed_features: torch.Tensor
    singular_values: torch.Tensor
    selected_rank: int
    explained_variance_retained: float
    policy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'projected_shape': tuple(int(value) for value in self.projected_features.shape),
            'basis_shape': tuple(int(value) for value in self.basis.shape),
            'selected_rank': int(self.selected_rank),
            'explained_variance_retained': float(self.explained_variance_retained),
            'policy': str(self.policy),
            **self.metadata,
        }


@dataclass(frozen=True)
class ForecastHeadResult:
    forecast: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'forecast_shape': tuple(int(value) for value in self.forecast.shape),
            **self.metadata,
        }


class ForecastingBoundaryAdapter:
    @staticmethod
    def from_input_data(
            input_data: InputData,
            *,
            forecast_horizon: int | None = None,
            device_policy: TensorDevicePolicy | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ForecastTensorBatch:
        resolved_horizon = (
            int(forecast_horizon)
            if forecast_horizon is not None
            else int(input_data.task.task_params.forecast_length)
        )
        return series_to_forecast_tensor_batch(
            input_data.features,
            forecast_horizon=resolved_horizon,
            device_policy=device_policy,
            idx=getattr(input_data, 'idx', None),
            metadata=metadata,
        )

    @staticmethod
    def to_output_data(
            input_data: InputData,
            forecast: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> OutputData:
        if OutputData is None:  # pragma: no cover - lightweight test envs
            return type('OutputData', (), {'predict': forecast, 'data_type': DataTypesEnum.table})()
        prediction = np.asarray(
            forecast.detach().cpu().numpy() if isinstance(forecast, torch.Tensor) else forecast,
            dtype=float,
        )
        return OutputData(
            idx=input_data.idx,
            features=input_data.features,
            predict=prediction,
            target=getattr(input_data, 'target', None),
            task=input_data.task,
            data_type=DataTypesEnum.table,
            supplementary_data=input_data.supplementary_data,
        )


class ForecastingRuntimeAdapter:
    capability = ForecastingOperationCapability(
        name='forecasting_runtime',
        stage='runtime',
        tensor_native=True,
        supports_multivariate=True,
        supports_tuning=True,
        supports_cuda=True,
    )

    def __init__(self, device_policy: TensorDevicePolicy | None = None):
        self.device_policy = device_policy or TensorDevicePolicy()

    def make_batch(
            self,
            time_series: Sequence[float] | np.ndarray | torch.Tensor,
            *,
            forecast_horizon: int,
            idx: Sequence[int] | np.ndarray | torch.Tensor | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ForecastTensorBatch:
        return series_to_forecast_tensor_batch(
            time_series,
            forecast_horizon=forecast_horizon,
            device_policy=self.device_policy,
            idx=idx,
            metadata=metadata,
        )


@dataclass
class RidgeForecastingHead:
    alpha: float = 1.0
    device_policy: TensorDevicePolicy = field(default_factory=TensorDevicePolicy)
    weights_: torch.Tensor | None = None
    intercept_: torch.Tensor | None = None
    input_dim_: int | None = None
    output_dim_: int | None = None

    def fit(self, features: torch.Tensor, target: torch.Tensor) -> 'RidgeForecastingHead':
        X = ensure_tensor_2d(features, self.device_policy)
        Y = ensure_tensor_2d(target, self.device_policy)
        design = torch.cat(
            [torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype), X],
            dim=1,
        )
        regularizer = torch.eye(design.shape[1], device=X.device, dtype=X.dtype) * float(self.alpha)
        regularizer[0, 0] = 0.0
        lhs = design.T @ design + regularizer
        rhs = design.T @ Y
        try:
            solution = torch.linalg.solve(lhs, rhs)
        except RuntimeError:  # pragma: no cover - defensive fallback
            solution = torch.linalg.pinv(lhs) @ rhs
        self.intercept_ = solution[0]
        self.weights_ = solution[1:]
        self.input_dim_ = int(X.shape[1])
        self.output_dim_ = int(Y.shape[1])
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights_ is None or self.intercept_ is None:
            raise ValueError('RidgeForecastingHead is not fitted.')
        X = ensure_tensor_2d(features, self.device_policy).to(device=self.weights_.device, dtype=self.weights_.dtype)
        return X @ self.weights_ + self.intercept_

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            'alpha': float(self.alpha),
            'input_dim': self.input_dim_,
            'output_dim': self.output_dim_,
            'device': str(self.weights_.device) if self.weights_ is not None else str(self.device_policy.device),
        }


class _ForecastingMLP(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Sequence[int],
                 activation: str = 'relu'):
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_dim = int(input_dim)
        activation_name = str(activation).lower()
        activation_factory = {
            'relu': torch.nn.ReLU,
            'gelu': torch.nn.GELU,
            'tanh': torch.nn.Tanh,
            'elu': torch.nn.ELU,
        }.get(activation_name, torch.nn.ReLU)
        for hidden_dim in hidden_dims:
            resolved_hidden_dim = int(max(1, hidden_dim))
            layers.append(torch.nn.Linear(current_dim, resolved_hidden_dim))
            layers.append(activation_factory())
            current_dim = resolved_hidden_dim
        layers.append(torch.nn.Linear(current_dim, int(output_dim)))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@dataclass
class MLPForecastingHead:
    hidden_dims: tuple[int, ...] = (64, 32)
    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    activation: str = 'relu'
    random_seed: int = 42
    device_policy: TensorDevicePolicy = field(default_factory=TensorDevicePolicy)
    network_: torch.nn.Module | None = None
    input_dim_: int | None = None
    output_dim_: int | None = None
    feature_mean_: torch.Tensor | None = None
    feature_std_: torch.Tensor | None = None
    target_mean_: torch.Tensor | None = None
    target_std_: torch.Tensor | None = None
    final_loss_: float | None = None

    def _normalize_features(self, values: torch.Tensor) -> torch.Tensor:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError('MLPForecastingHead is not fitted.')
        return (values - self.feature_mean_) / self.feature_std_

    def _denormalize_target(self, values: torch.Tensor) -> torch.Tensor:
        if self.target_mean_ is None or self.target_std_ is None:
            raise ValueError('MLPForecastingHead is not fitted.')
        return values * self.target_std_ + self.target_mean_

    def fit(self, features: torch.Tensor, target: torch.Tensor) -> 'MLPForecastingHead':
        X = ensure_tensor_2d(features, self.device_policy)
        Y = ensure_tensor_2d(target, self.device_policy)
        self.input_dim_ = int(X.shape[1])
        self.output_dim_ = int(Y.shape[1])
        self.feature_mean_ = torch.mean(X, dim=0, keepdim=True)
        self.feature_std_ = torch.std(X, dim=0, keepdim=True, unbiased=False)
        self.feature_std_ = torch.where(self.feature_std_ < 1e-6, torch.ones_like(self.feature_std_), self.feature_std_)
        self.target_mean_ = torch.mean(Y, dim=0, keepdim=True)
        self.target_std_ = torch.std(Y, dim=0, keepdim=True, unbiased=False)
        self.target_std_ = torch.where(self.target_std_ < 1e-6, torch.ones_like(self.target_std_), self.target_std_)
        normalized_x = self._normalize_features(X)
        normalized_y = (Y - self.target_mean_) / self.target_std_

        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(int(self.random_seed))
        try:
            network = _ForecastingMLP(
                input_dim=self.input_dim_,
                output_dim=self.output_dim_,
                hidden_dims=tuple(int(max(1, value)) for value in self.hidden_dims),
                activation=self.activation,
            ).to(device=X.device, dtype=X.dtype)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = torch.nn.MSELoss()
        network.train()
        for _ in range(int(max(1, self.epochs))):
            optimizer.zero_grad(set_to_none=True)
            prediction = network(normalized_x)
            loss = loss_fn(prediction, normalized_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

        self.network_ = network
        self.final_loss_ = float(loss.detach().cpu().item())
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.network_ is None:
            raise ValueError('MLPForecastingHead is not fitted.')
        X = ensure_tensor_2d(features, self.device_policy).to(
            device=self.feature_mean_.device,
            dtype=self.feature_mean_.dtype,
        )
        self.network_.eval()
        with torch.no_grad():
            normalized_prediction = self.network_(self._normalize_features(X))
        return self._denormalize_target(normalized_prediction)

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            'head_policy': 'mlp',
            'hidden_dims': tuple(int(value) for value in self.hidden_dims),
            'epochs': int(self.epochs),
            'learning_rate': float(self.learning_rate),
            'weight_decay': float(self.weight_decay),
            'activation': str(self.activation),
            'input_dim': self.input_dim_,
            'output_dim': self.output_dim_,
            'final_loss': self.final_loss_,
            'device': str(self.feature_mean_.device) if self.feature_mean_ is not None else str(
                self.device_policy.device),
        }


@dataclass
class WeightedAverageHead:
    device_policy: TensorDevicePolicy = field(default_factory=TensorDevicePolicy)
    weights_: torch.Tensor | None = None

    def fit(self, branch_forecasts: torch.Tensor, target: torch.Tensor) -> 'WeightedAverageHead':
        forecasts = ensure_tensor_2d(branch_forecasts, self.device_policy)
        if forecasts.shape[0] < forecasts.shape[1]:
            design = forecasts.T
        else:
            design = forecasts
        target_tensor = ensure_tensor_2d(target, self.device_policy).reshape(-1, 1)
        if target_tensor.shape[0] != design.shape[0]:
            if target_tensor.shape[1] == design.shape[0]:
                target_tensor = target_tensor.T
            else:
                raise ValueError('Branch forecasts and target length must match.')
        weights = torch.linalg.pinv(design) @ target_tensor
        weights = torch.clamp(weights.reshape(-1), min=0.0)
        if float(torch.sum(weights)) <= 1e-8:
            weights = torch.ones_like(weights) / max(1, len(weights))
        else:
            weights = weights / torch.sum(weights)
        self.weights_ = weights
        return self

    def predict(self, branch_forecasts: torch.Tensor) -> torch.Tensor:
        if self.weights_ is None:
            raise ValueError('WeightedAverageHead is not fitted.')
        forecasts = ensure_tensor_2d(branch_forecasts, self.device_policy)
        if forecasts.shape[0] == len(self.weights_):
            return self.weights_ @ forecasts
        return forecasts @ self.weights_

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            'weights': [] if self.weights_ is None else [float(value) for value in
                                                         self.weights_.detach().cpu().tolist()]
        }


def _normalize_series_array(time_series: Sequence[float] | np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(time_series, torch.Tensor):
        normalized = time_series.detach().cpu().numpy()
    else:
        normalized = np.asarray(time_series, dtype=float)
    if normalized.ndim == 1:
        return normalized.reshape(-1, 1)
    if normalized.ndim != 2:
        raise ValueError('time_series must be 1D or 2D.')
    if normalized.shape[0] < normalized.shape[1] and normalized.shape[0] <= 8:
        normalized = normalized.T
    return normalized.astype(float, copy=False)


def ensure_tensor_2d(
        values: Sequence[float] | np.ndarray | torch.Tensor,
        device_policy: TensorDevicePolicy | None = None,
) -> torch.Tensor:
    resolved_policy = device_policy or TensorDevicePolicy()
    if isinstance(values, torch.Tensor):
        tensor = values
    else:
        tensor = torch.as_tensor(values, dtype=resolved_policy.resolve_dtype())
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim != 2:
        raise ValueError('Expected a 2D tensor-compatible value.')
    return tensor.to(device=resolved_policy.resolve_device(), dtype=resolved_policy.resolve_dtype())


def series_to_forecast_tensor_batch(
        time_series: Sequence[float] | np.ndarray | torch.Tensor,
        *,
        forecast_horizon: int,
        device_policy: TensorDevicePolicy | None = None,
        idx: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
) -> ForecastTensorBatch:
    resolved_policy = device_policy or TensorDevicePolicy()
    normalized = _normalize_series_array(time_series)
    history = torch.as_tensor(
        normalized,
        dtype=resolved_policy.resolve_dtype(),
        device=resolved_policy.resolve_device(),
    )
    idx_tensor = None
    if idx is not None:
        idx_tensor = torch.as_tensor(idx, device=resolved_policy.resolve_device())
    return ForecastTensorBatch(
        history=history,
        target=None,
        forecast_horizon=int(forecast_horizon),
        idx=idx_tensor,
        metadata=dict(metadata or {}),
    )


def resolve_window_size(
        series_length: int,
        forecast_horizon: int,
        *,
        window_size: int | None = None,
        window_size_percent: float | None = None,
        min_ratio: float = 0.10,
        max_ratio: float = 0.35,
) -> int:
    max_window = max(2, series_length - forecast_horizon)
    if window_size is not None:
        return int(max(2, min(int(window_size), max_window)))
    if window_size_percent is not None:
        percent_window = round(series_length * 0.01 * float(window_size_percent))
        return int(max(2, min(percent_window, max_window)))
    candidate = int(round(series_length * (min_ratio + max_ratio) / 2.0))
    return int(max(forecast_horizon + 1, min(candidate, max_window)))


def resolve_stride(window_size: int, stride: int | None = None) -> int:
    requested = 1 if stride is None else int(stride)
    return int(max(1, min(requested, max(1, window_size // 2))))


def build_hankel_trajectory_transform(
        batch: ForecastTensorBatch,
        *,
        window_size: int,
        stride: int = 1,
) -> TrajectoryTransformResult:
    history = batch.history
    series_length, channel_count = history.shape
    resolved_window = resolve_window_size(
        series_length=series_length,
        forecast_horizon=batch.forecast_horizon,
        window_size=window_size,
    )
    resolved_stride = resolve_stride(resolved_window, stride)
    max_start = series_length - resolved_window - batch.forecast_horizon
    if max_start < 0:
        raise ValueError(
            f'Series length {series_length} is insufficient for window_size={resolved_window} '
            f'and forecast_horizon={batch.forecast_horizon}.'
        )

    feature_rows = []
    target_rows = []
    for start in range(0, max_start + 1, resolved_stride):
        feature_window = history[start:start + resolved_window]
        target_window = history[start + resolved_window:start + resolved_window + batch.forecast_horizon]
        feature_rows.append(feature_window.reshape(-1))
        target_rows.append(target_window.reshape(-1))
    if not feature_rows:
        raise ValueError('Hankelisation produced no supervised windows.')

    latest_features = history[-resolved_window:].reshape(1, -1)
    features = torch.stack(feature_rows)
    target = torch.stack(target_rows)
    return TrajectoryTransformResult(
        features=features,
        target=target,
        latest_features=latest_features,
        window_size=resolved_window,
        stride=resolved_stride,
        forecast_horizon=batch.forecast_horizon,
        channel_count=channel_count,
        metadata={
            'series_length': int(series_length),
            'n_samples': int(features.shape[0]),
            'feature_dim': int(features.shape[1]),
            'target_dim': int(target.shape[1]),
            'representation': 'supervised_hankel',
        },
    )


def compute_svd_decomposition(
        features: torch.Tensor,
        *,
        strategy: str = 'full',
) -> DecompositionResult:
    matrix = ensure_tensor_2d(features)
    U, singular_values, Vh = torch.linalg.svd(matrix, full_matrices=False)
    projected = U * singular_values
    basis = Vh.T
    return DecompositionResult(
        projected_features=projected,
        basis=basis,
        singular_values=singular_values,
        strategy=str(strategy),
        input_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        metadata={'approximation': 'exact' if strategy == 'full' else 'compat_full_svd'},
    )


def compute_randomized_svd_decomposition(
        features: torch.Tensor,
        *,
        n_oversamples: int = 5,
) -> DecompositionResult:
    result = compute_svd_decomposition(features, strategy='randomized')
    metadata = dict(result.metadata)
    metadata['n_oversamples'] = int(n_oversamples)
    return DecompositionResult(
        projected_features=result.projected_features,
        basis=result.basis,
        singular_values=result.singular_values,
        strategy='randomized',
        input_shape=result.input_shape,
        metadata=metadata,
    )


def compute_tensor_decomposition(
        features: torch.Tensor,
        *,
        unfolding_strategy: str = 'channels_last',
) -> DecompositionResult:
    matrix = ensure_tensor_2d(features)
    result = compute_svd_decomposition(matrix, strategy='tensor_compat')
    metadata = dict(result.metadata)
    metadata['unfolding_strategy'] = str(unfolding_strategy)
    return DecompositionResult(
        projected_features=result.projected_features,
        basis=result.basis,
        singular_values=result.singular_values,
        strategy='tensor_compat',
        input_shape=result.input_shape,
        metadata=metadata,
    )


def truncate_decomposition_rank(
        decomposition: DecompositionResult,
        *,
        rank: int | None = None,
        explained_variance: float = 0.95,
        policy: str = 'explained_variance',
        expert_rank: int | None = None,
        min_rank: int = 1,
) -> RankTruncationResult:
    singular_values = decomposition.singular_values
    min_dim = min(int(decomposition.projected_features.shape[0]), int(decomposition.projected_features.shape[1]))
    if policy == 'expert':
        selected_rank = expert_rank if expert_rank is not None else rank
    elif policy == 'statistical':
        median = torch.median(torch.abs(singular_values))
        threshold = float(median) * 1.5
        selected_rank = int(torch.sum(torch.abs(singular_values) >= threshold).item())
    else:
        energy = singular_values ** 2
        total = float(torch.sum(energy).item())
        if rank is not None:
            selected_rank = int(rank)
        elif total <= 0:
            selected_rank = min_dim
        else:
            cumulative = torch.cumsum(energy, dim=0) / total
            selected_rank = int(
                torch.searchsorted(cumulative, torch.tensor(explained_variance, device=cumulative.device)).item() + 1)
    selected_rank = int(max(min_rank, min(selected_rank, int(singular_values.shape[0]), min_dim)))
    basis = decomposition.basis[:, :selected_rank]
    projected = decomposition.projected_features[:, :selected_rank]
    reconstructed = projected @ basis.T
    retained = float(
        torch.sum(singular_values[:selected_rank] ** 2).item() / max(float(torch.sum(singular_values ** 2).item()),
                                                                     1e-12)
    )
    return RankTruncationResult(
        projected_features=projected,
        basis=basis,
        reconstructed_features=reconstructed,
        singular_values=singular_values,
        selected_rank=selected_rank,
        explained_variance_retained=retained,
        policy=str(policy),
        metadata={'explained_variance_target': float(explained_variance)},
    )


def project_features(features: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    X = ensure_tensor_2d(features)
    return X @ basis


def inverse_project_features(projected_features: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    Z = ensure_tensor_2d(projected_features)
    return Z @ basis.T


def split_forecasting_batch(
        batch: ForecastTensorBatch,
        split_spec: ForecastingSplitSpec | None = None,
) -> tuple[ForecastTensorBatch, torch.Tensor]:
    spec = split_spec or ForecastingSplitSpec(validation_horizon=batch.forecast_horizon)
    validation_horizon = int(spec.validation_horizon or batch.forecast_horizon)
    if spec.kind is not ForecastingSplitKind.HOLDOUT:
        raise NotImplementedError(f'Only holdout split is implemented in wave 1, got {spec.kind}.')
    min_train = int(spec.min_train_length or max(batch.forecast_horizon + 2, validation_horizon * 2))
    if batch.series_length <= min_train + validation_horizon:
        raise ValueError('Series is too short for the requested holdout split.')
    train_history = batch.history[:-validation_horizon]
    validation_target = batch.history[-validation_horizon:].reshape(-1)
    train_batch = ForecastTensorBatch(
        history=train_history,
        target=None,
        forecast_horizon=batch.forecast_horizon,
        idx=None if batch.idx is None else batch.idx[:-validation_horizon],
        metadata={**batch.metadata, 'split_kind': spec.kind.value, 'validation_horizon': validation_horizon},
    )
    return train_batch, validation_target


def evaluate_forecast(
        y_true: Sequence[float] | np.ndarray | torch.Tensor,
        y_pred: Sequence[float] | np.ndarray | torch.Tensor,
        *,
        metric_name: str = 'rmse',
) -> ForecastingEvaluationResult:
    truth = torch.as_tensor(y_true, dtype=torch.float32).reshape(-1)
    pred = torch.as_tensor(y_pred, dtype=torch.float32).reshape(-1)
    if truth.shape[0] != pred.shape[0]:
        raise ValueError('y_true and y_pred must have the same length.')
    error = pred - truth
    if metric_name == 'mae':
        per_horizon = torch.abs(error)
        value = float(torch.mean(per_horizon).item())
    else:
        per_horizon = torch.sqrt(torch.clamp(error ** 2, min=0.0))
        value = float(torch.sqrt(torch.mean(error ** 2)).item())
        metric_name = 'rmse'
    return ForecastingEvaluationResult(
        metric_name=metric_name,
        metric_value=value,
        per_horizon_metrics=tuple(float(item) for item in per_horizon.detach().cpu().tolist()),
        metadata={'n_horizons': int(len(per_horizon))},
    )


def capability_to_dict(capability: ForecastingOperationCapability) -> dict[str, Any]:
    return asdict(capability)

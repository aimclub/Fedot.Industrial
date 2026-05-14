from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import torch

try:  # pragma: no cover - progress bar is optional in lightweight envs
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from .progress_policy import ForecastingProgressPolicy, resolve_forecasting_progress_policy

try:  # pragma: no cover - optional FEDOT runtime in lightweight envs
    from fedot.core.data.data import InputData, OutputData
    from fedot.core.repository.dataset_types import DataTypesEnum
except Exception:  # pragma: no cover
    InputData = OutputData = None

    class DataTypesEnum:  # type: ignore[override]
        table = 'table'
        ts = 'ts'


class ForecastingSplitKind(str, Enum):
    """Supported temporal validation split strategies for forecasting runtime."""

    HOLDOUT = 'holdout'
    TIME_SERIES_SPLIT = 'time_series_split'
    EXPANDING_WINDOW = 'expanding_window'
    ROLLING_WINDOW = 'rolling_window'
    ROLLING_ORIGIN = 'rolling_origin'
    BLOCKED = 'blocked'


@dataclass(frozen=True)
class TensorDevicePolicy:
    """Resolve tensor device and dtype for forecasting runtime operations."""

    device: str = 'auto'
    dtype: str = 'float32'

    def resolve_device(self) -> torch.device:
        """Return the concrete torch device, preferring CUDA when policy is auto."""
        requested = str(self.device).lower()
        if requested == 'auto':
            return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if requested.startswith('cuda') and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device('cpu')

    def resolve_dtype(self) -> torch.dtype:
        """Return the torch dtype requested by the policy with float32 fallback."""
        try:
            return getattr(torch, str(self.dtype))
        except AttributeError:  # pragma: no cover - defensive fallback
            return torch.float32


@dataclass(frozen=True)
class ForecastTensorBatch:
    """Canonical tensor-native payload passed through forecasting primitives."""

    history: torch.Tensor
    target: torch.Tensor | None
    forecast_horizon: int
    idx: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        """Return the device where historical values are stored."""
        return self.history.device

    @property
    def channel_count(self) -> int:
        """Return the number of channels in the canonical history tensor."""
        return int(self.history.shape[1])

    @property
    def series_length(self) -> int:
        """Return the temporal length of the canonical history tensor."""
        return int(self.history.shape[0])

    def to(self, device_policy: TensorDevicePolicy) -> 'ForecastTensorBatch':
        """Move history, target and index tensors according to a device policy."""
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
        """Serialize lightweight batch metadata for diagnostics and artifacts."""
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
    """Describe runtime capabilities exposed by a forecasting primitive or adapter."""

    name: str
    stage: str
    tensor_native: bool = True
    supports_multivariate: bool = True
    supports_tuning: bool = True
    supports_cuda: bool = True


@dataclass(frozen=True)
class ForecastingSplitSpec:
    """Configuration for explicit temporal validation splits."""

    kind: ForecastingSplitKind = ForecastingSplitKind.HOLDOUT
    validation_horizon: int | None = None
    min_train_length: int | None = None
    n_splits: int | None = None
    test_size: int | None = None
    gap: int = 0
    max_train_size: int | None = None
    initial_window: int | None = None
    step_length: int | None = None


@dataclass(frozen=True)
class ForecastingFoldSplit:
    """One temporal validation fold with a train batch and horizon target."""

    train_batch: ForecastTensorBatch
    validation_target: torch.Tensor
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fold boundaries and split metadata for benchmark artifacts."""
        return {
            'fold_index': int(self.fold_index),
            'train_start': int(self.train_start),
            'train_end': int(self.train_end),
            'test_start': int(self.test_start),
            'test_end': int(self.test_end),
            'train_length': int(self.train_end - self.train_start),
            'test_length': int(self.test_end - self.test_start),
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class ForecastingEvaluationResult:
    """Metric value with per-horizon diagnostics for a forecast vector."""

    metric_name: str
    metric_value: float
    per_horizon_metrics: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metric data in a JSON-friendly format."""
        return {
            'metric_name': self.metric_name,
            'metric_value': float(self.metric_value),
            'per_horizon_metrics': [float(value) for value in self.per_horizon_metrics],
            **self.metadata,
        }


@dataclass(frozen=True)
class TrajectoryTransformResult:
    """Output of a trajectory transform such as hankelisation."""

    features: torch.Tensor
    target: torch.Tensor | None
    latest_features: torch.Tensor
    window_size: int
    stride: int
    forecast_horizon: int
    channel_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize trajectory shapes and transform parameters."""
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
    """Low-rank decomposition payload shared by SVD-like primitives."""

    projected_features: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    strategy: str
    input_shape: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize decomposition shapes and strategy metadata."""
        return {
            'projected_shape': tuple(int(value) for value in self.projected_features.shape),
            'basis_shape': tuple(int(value) for value in self.basis.shape),
            'input_shape': tuple(int(value) for value in self.input_shape),
            'strategy': str(self.strategy),
            **self.metadata,
        }


@dataclass(frozen=True)
class RankTruncationResult:
    """Rank-truncated representation plus reconstruction diagnostics."""

    projected_features: torch.Tensor
    basis: torch.Tensor
    reconstructed_features: torch.Tensor
    singular_values: torch.Tensor
    selected_rank: int
    explained_variance_retained: float
    policy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize selected-rank diagnostics and output shapes."""
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
    """Forecast head output payload for primitive graph execution."""

    forecast: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize forecast tensor shape and head metadata."""
        return {
            'forecast_shape': tuple(int(value) for value in self.forecast.shape),
            **self.metadata,
        }


class ForecastingBoundaryAdapter:
    """Convert FEDOT boundary objects to and from tensor-native runtime payloads."""

    @staticmethod
    def from_input_data(
            input_data: InputData,
            *,
            forecast_horizon: int | None = None,
            device_policy: TensorDevicePolicy | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ForecastTensorBatch:
        """Build a ForecastTensorBatch from FEDOT InputData."""
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
        """Wrap a forecast vector back into FEDOT OutputData."""
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
    """Small convenience adapter for creating runtime batches from raw series."""

    capability = ForecastingOperationCapability(
        name='forecasting_runtime',
        stage='runtime',
        tensor_native=True,
        supports_multivariate=True,
        supports_tuning=True,
        supports_cuda=True,
    )

    def __init__(self, device_policy: TensorDevicePolicy | None = None):
        """Store the device policy used by all batches produced by this adapter."""
        self.device_policy = device_policy or TensorDevicePolicy()

    def make_batch(
            self,
            time_series: Sequence[float] | np.ndarray | torch.Tensor,
            *,
            forecast_horizon: int,
            idx: Sequence[int] | np.ndarray | torch.Tensor | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ForecastTensorBatch:
        """Convert a raw time series into a ForecastTensorBatch."""
        return series_to_forecast_tensor_batch(
            time_series,
            forecast_horizon=forecast_horizon,
            device_policy=self.device_policy,
            idx=idx,
            metadata=metadata,
        )


@dataclass
class RidgeForecastingHead:
    """Closed-form ridge regression head for lagged forecasting features."""

    alpha: float = 1.0
    device_policy: TensorDevicePolicy = field(default_factory=TensorDevicePolicy)
    weights_: torch.Tensor | None = None
    intercept_: torch.Tensor | None = None
    input_dim_: int | None = None
    output_dim_: int | None = None

    def fit(self, features: torch.Tensor, target: torch.Tensor) -> 'RidgeForecastingHead':
        """Fit ridge weights from supervised trajectory features to horizon targets."""
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
        """Predict horizon vectors for trajectory features using fitted weights."""
        if self.weights_ is None or self.intercept_ is None:
            raise ValueError('RidgeForecastingHead is not fitted.')
        X = ensure_tensor_2d(features, self.device_policy).to(device=self.weights_.device, dtype=self.weights_.dtype)
        return X @ self.weights_ + self.intercept_

    def get_diagnostics(self) -> dict[str, Any]:
        """Return fitted ridge dimensions and runtime device metadata."""
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
        """Create a feed-forward network used by MLPForecastingHead."""
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
        """Run a forward pass through the internal sequential network."""
        return self.network(values)


def build_scaled_mlp_hidden_dims(
        depth: int,
        *,
        base_hidden_dim: int = 512,
        min_hidden_dim: int = 8,
) -> tuple[int, ...]:
    """Build monotonically shrinking hidden dimensions for an MLP depth."""
    resolved_depth = int(max(1, depth))
    current_dim = int(max(min_hidden_dim, base_hidden_dim))
    resolved_dims: list[int] = []
    for _ in range(resolved_depth):
        resolved_dims.append(int(max(min_hidden_dim, current_dim)))
        current_dim = max(min_hidden_dim, current_dim // 2)
    return tuple(resolved_dims)


@dataclass
class MLPForecastingHead:
    """Trainable MLP forecast head with normalization, scheduler and early stopping."""

    hidden_dims: tuple[int, ...] | None = None
    depth: int = 2
    base_hidden_dim: int = 512
    min_hidden_dim: int = 8
    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    activation: str = 'relu'
    validation_fraction: float = 0.2
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-5
    scheduler_patience: int = 6
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-5
    random_seed: int = 42
    show_progress: bool | None = None
    progress_policy: ForecastingProgressPolicy | dict[str, Any] | bool | None = None
    device_policy: TensorDevicePolicy = field(default_factory=TensorDevicePolicy)
    network_: torch.nn.Module | None = None
    input_dim_: int | None = None
    output_dim_: int | None = None
    feature_mean_: torch.Tensor | None = None
    feature_std_: torch.Tensor | None = None
    target_mean_: torch.Tensor | None = None
    target_std_: torch.Tensor | None = None
    final_loss_: float | None = None
    best_validation_loss_: float | None = None
    best_epoch_: int | None = None
    stopped_early_: bool = False
    resolved_hidden_dims_: tuple[int, ...] = field(default_factory=tuple)
    final_learning_rate_: float | None = None

    def _resolve_hidden_dims(self) -> tuple[int, ...]:
        if self.hidden_dims:
            return tuple(int(max(1, value)) for value in self.hidden_dims)
        return build_scaled_mlp_hidden_dims(
            int(max(1, self.depth)),
            base_hidden_dim=int(max(1, self.base_hidden_dim)),
            min_hidden_dim=int(max(1, self.min_hidden_dim)),
        )

    @staticmethod
    def _compute_tensor_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.mean(values, dim=0, keepdim=True)
        std = torch.std(values, dim=0, keepdim=True, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return mean, std

    def _split_train_validation(
            self,
            features: torch.Tensor,
            target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_samples = int(features.shape[0])
        if n_samples <= 2:
            return features, target, features, target
        validation_fraction = float(min(0.5, max(0.05, self.validation_fraction)))
        validation_size = int(max(1, round(n_samples * validation_fraction)))
        validation_size = min(validation_size, n_samples - 1)
        train_size = n_samples - validation_size
        if train_size <= 0:
            return features, target, features, target
        return (
            features[:train_size],
            target[:train_size],
            features[train_size:],
            target[train_size:],
        )

    def _normalize_features(self, values: torch.Tensor) -> torch.Tensor:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError('MLPForecastingHead is not fitted.')
        return (values - self.feature_mean_) / self.feature_std_

    def _normalize_target(self, values: torch.Tensor) -> torch.Tensor:
        if self.target_mean_ is None or self.target_std_ is None:
            raise ValueError('MLPForecastingHead is not fitted.')
        return (values - self.target_mean_) / self.target_std_

    def _denormalize_target(self, values: torch.Tensor) -> torch.Tensor:
        if self.target_mean_ is None or self.target_std_ is None:
            raise ValueError('MLPForecastingHead is not fitted.')
        return values * self.target_std_ + self.target_mean_

    def _initialize_data_statistics(self, features: torch.Tensor, target: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor]:
        self.input_dim_ = int(features.shape[1])
        self.output_dim_ = int(target.shape[1])
        self.feature_mean_, self.feature_std_ = self._compute_tensor_statistics(features)
        self.target_mean_, self.target_std_ = self._compute_tensor_statistics(target)
        normalized_x = self._normalize_features(features)
        normalized_y = self._normalize_target(target)
        return normalized_x, normalized_y

    def _prepare_fit_data(self, features: torch.Tensor, target: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        X = ensure_tensor_2d(features, self.device_policy)
        Y = ensure_tensor_2d(target, self.device_policy)
        normalized_x, normalized_y = self._initialize_data_statistics(X, Y)
        train_x, train_y, validation_x, validation_y = self._split_train_validation(normalized_x, normalized_y)
        return X, Y, train_x, train_y, validation_x, validation_y

    def _build_network(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.nn.Module, tuple[int, ...]]:
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(int(self.random_seed))
        resolved_hidden_dims = self._resolve_hidden_dims()
        try:
            network = _ForecastingMLP(
                input_dim=self.input_dim_,
                output_dim=self.output_dim_,
                hidden_dims=resolved_hidden_dims,
                activation=self.activation,
            ).to(device=device, dtype=dtype)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        return network, resolved_hidden_dims

    def _build_optimizer(self, network: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            network.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(self.scheduler_factor),
            patience=int(max(1, self.scheduler_patience)),
            min_lr=float(self.scheduler_min_lr),
        )

    def _build_loss_fn(self):
        return torch.nn.MSELoss()

    def _build_epoch_iterator(self, resolved_progress_policy: ForecastingProgressPolicy):
        return tqdm(
            range(int(max(1, self.epochs))),
            **resolved_progress_policy.tqdm_kwargs(
                scope='head_training',
                desc='MLP head fit',
                unit='epoch',
            ),
        )

    def _should_update_epoch_postfix(self, epoch_index: int) -> bool:
        resolved_epochs = int(max(1, self.epochs))
        return (
            epoch_index == 0
            or (epoch_index + 1) % max(1, resolved_epochs // 10) == 0
            or epoch_index + 1 == resolved_epochs
        )

    def _run_training_epoch(
            self,
            *,
            network: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            validation_x: torch.Tensor,
            validation_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        train_prediction = network(train_x)
        train_loss = loss_fn(train_prediction, train_y)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()
        network.eval()
        with torch.no_grad():
            validation_prediction = network(validation_x)
            validation_loss = loss_fn(validation_prediction, validation_y)
        network.train()
        return train_loss, validation_loss

    def _run_training_loop(
            self,
            *,
            network: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler,
            loss_fn,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            validation_x: torch.Tensor,
            validation_y: torch.Tensor,
            resolved_progress_policy: ForecastingProgressPolicy,
    ) -> tuple[torch.nn.Module, float, int, bool]:
        network.train()
        best_state_dict = copy.deepcopy(network.state_dict())
        best_validation_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        stopped_early = False
        epoch_iterator = self._build_epoch_iterator(resolved_progress_policy)

        for epoch_index in epoch_iterator:
            train_loss, validation_loss = self._run_training_epoch(
                network=network,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_x=train_x,
                train_y=train_y,
                validation_x=validation_x,
                validation_y=validation_y,
            )
            monitored_loss = float(validation_loss.detach().cpu().item())
            scheduler.step(validation_loss)
            if monitored_loss + float(self.early_stopping_min_delta) < best_validation_loss:
                best_validation_loss = monitored_loss
                best_epoch = int(epoch_index) + 1
                patience_counter = 0
                best_state_dict = copy.deepcopy(network.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= int(max(1, self.early_stopping_patience)):
                    stopped_early = True
            if resolved_progress_policy.show_postfix and hasattr(epoch_iterator, 'set_postfix') and \
                    self._should_update_epoch_postfix(int(epoch_index)):
                epoch_iterator.set_postfix(
                    train_loss=f'{float(train_loss.detach().cpu().item()):.5f}',
                    val_loss=f'{monitored_loss:.5f}',
                    lr=f"{float(optimizer.param_groups[0]['lr']):.2e}",
                )
            if stopped_early:
                break

        network.load_state_dict(best_state_dict)
        return network, float(best_validation_loss), int(best_epoch), bool(stopped_early)

    def _finalize_fit_state(
            self,
            *,
            network: torch.nn.Module,
            best_validation_loss: float,
            best_epoch: int,
            stopped_early: bool,
            resolved_hidden_dims: tuple[int, ...],
            optimizer: torch.optim.Optimizer,
    ) -> None:
        self.network_ = network
        self.final_loss_ = float(best_validation_loss)
        self.best_validation_loss_ = float(best_validation_loss)
        self.best_epoch_ = int(best_epoch)
        self.stopped_early_ = bool(stopped_early)
        self.resolved_hidden_dims_ = tuple(int(value) for value in resolved_hidden_dims)
        self.final_learning_rate_ = float(optimizer.param_groups[0]['lr'])

    def fit(self, features: torch.Tensor, target: torch.Tensor) -> 'MLPForecastingHead':
        """Fit the MLP head on supervised features and horizon targets."""
        resolved_progress_policy = resolve_forecasting_progress_policy(
            self.progress_policy,
            show_progress=self.show_progress,
        )
        X, _, train_x, train_y, validation_x, validation_y = self._prepare_fit_data(features, target)
        network, resolved_hidden_dims = self._build_network(device=X.device, dtype=X.dtype)
        optimizer = self._build_optimizer(network)
        scheduler = self._build_scheduler(optimizer)
        loss_fn = self._build_loss_fn()
        network, best_validation_loss, best_epoch, stopped_early = self._run_training_loop(
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            resolved_progress_policy=resolved_progress_policy,
        )
        self._finalize_fit_state(
            network=network,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
            resolved_hidden_dims=resolved_hidden_dims,
            optimizer=optimizer,
        )
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict denormalized horizon vectors with the fitted MLP network."""
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
        """Return training, architecture and device diagnostics for the MLP head."""
        return {
            'head_policy': 'mlp',
            'hidden_dims': tuple(int(value) for value in self.resolved_hidden_dims_),
            'depth': int(max(1, self.depth)),
            'base_hidden_dim': int(max(1, self.base_hidden_dim)),
            'min_hidden_dim': int(max(1, self.min_hidden_dim)),
            'epochs': int(self.epochs),
            'learning_rate': float(self.learning_rate),
            'weight_decay': float(self.weight_decay),
            'activation': str(self.activation),
            'validation_fraction': float(self.validation_fraction),
            'early_stopping_patience': int(max(1, self.early_stopping_patience)),
            'early_stopping_min_delta': float(self.early_stopping_min_delta),
            'scheduler_patience': int(max(1, self.scheduler_patience)),
            'scheduler_factor': float(self.scheduler_factor),
            'scheduler_min_lr': float(self.scheduler_min_lr),
            'input_dim': self.input_dim_,
            'output_dim': self.output_dim_,
            'final_loss': self.final_loss_,
            'best_validation_loss': self.best_validation_loss_,
            'best_epoch': self.best_epoch_,
            'stopped_early': bool(self.stopped_early_),
            'final_learning_rate': self.final_learning_rate_,
            'progress_policy': resolve_forecasting_progress_policy(
                self.progress_policy,
                show_progress=self.show_progress,
            ).to_dict(),
            'device': str(self.feature_mean_.device) if self.feature_mean_ is not None else str(
                self.device_policy.device),
        }


@dataclass
class WeightedAverageHead:
    """Constrained non-negative ensemble head for branch forecast weighting."""

    device_policy: TensorDevicePolicy = field(default_factory=TensorDevicePolicy)
    weights_: torch.Tensor | None = None

    def fit(self, branch_forecasts: torch.Tensor, target: torch.Tensor) -> 'WeightedAverageHead':
        """Fit branch weights against a validation horizon target."""
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
        """Combine branch forecasts using the fitted normalized weights."""
        if self.weights_ is None:
            raise ValueError('WeightedAverageHead is not fitted.')
        forecasts = ensure_tensor_2d(branch_forecasts, self.device_policy)
        if forecasts.shape[0] == len(self.weights_):
            return self.weights_ @ forecasts
        return forecasts @ self.weights_

    def get_diagnostics(self) -> dict[str, Any]:
        """Return learned branch weights in a JSON-friendly form."""
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
    """Normalize array-like values to a 2D tensor on the requested device."""
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
    """Create the canonical ForecastTensorBatch from raw series values."""
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
    """Resolve and validate a trajectory window size for a series length."""
    max_window = max(2, series_length - forecast_horizon)
    if window_size is not None:
        return int(max(2, min(int(window_size), max_window)))
    if window_size_percent is not None:
        percent_window = round(series_length * 0.01 * float(window_size_percent))
        return int(max(2, min(percent_window, max_window)))
    candidate = int(round(series_length * (min_ratio + max_ratio) / 2.0))
    return int(max(forecast_horizon + 1, min(candidate, max_window)))


def resolve_stride(window_size: int, stride: int | None = None) -> int:
    """Clamp trajectory stride to a safe positive value for the window size."""
    requested = 1 if stride is None else int(stride)
    return int(max(1, min(requested, max(1, window_size // 2))))


def build_hankel_trajectory_transform(
        batch: ForecastTensorBatch,
        *,
        window_size: int,
        stride: int = 1,
) -> TrajectoryTransformResult:
    """Build supervised Hankel windows and horizon targets from a tensor batch."""
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
    """Compute an exact SVD-based feature decomposition."""
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
    """Compute the randomized-SVD compatible decomposition payload."""
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
    """Compute a tensor-compatible decomposition through an explicit unfolding policy."""
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
    """Select a low-rank approximation according to rank or variance policy."""
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
    """Project feature rows into a decomposition basis."""
    X = ensure_tensor_2d(features)
    return X @ basis


def inverse_project_features(projected_features: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Reconstruct feature rows from projected coordinates and a basis."""
    Z = ensure_tensor_2d(projected_features)
    return Z @ basis.T


def _resolve_split_horizon(batch: ForecastTensorBatch, spec: ForecastingSplitSpec) -> int:
    return int(spec.test_size or spec.validation_horizon or batch.forecast_horizon)


def _resolve_min_train_length(batch: ForecastTensorBatch,
                              spec: ForecastingSplitSpec,
                              validation_horizon: int) -> int:
    return int(spec.min_train_length or max(batch.forecast_horizon + 2, validation_horizon * 2))


def _resolve_target_fold_count(max_possible_folds: int, requested_splits: int | None) -> int:
    if max_possible_folds < 2:
        return 0
    requested = int(requested_splits) if requested_splits is not None else 10
    target = max(10, requested)
    return int(min(max_possible_folds, target))


def _build_fold_split(
        batch: ForecastTensorBatch,
        *,
        spec: ForecastingSplitSpec,
        fold_index: int,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
        validation_horizon: int,
) -> ForecastingFoldSplit:
    train_history = batch.history[train_start:train_end]
    validation_target = batch.history[test_start:test_end].reshape(-1)
    train_idx = None if batch.idx is None else batch.idx[train_start:train_end]
    train_batch = ForecastTensorBatch(
        history=train_history,
        target=None,
        forecast_horizon=batch.forecast_horizon,
        idx=train_idx,
        metadata={
            **batch.metadata,
            'split_kind': spec.kind.value,
            'validation_horizon': int(validation_horizon),
            'fold_index': int(fold_index),
            'train_start': int(train_start),
            'train_end': int(train_end),
            'test_start': int(test_start),
            'test_end': int(test_end),
        },
    )
    return ForecastingFoldSplit(
        train_batch=train_batch,
        validation_target=validation_target,
        fold_index=int(fold_index),
        train_start=int(train_start),
        train_end=int(train_end),
        test_start=int(test_start),
        test_end=int(test_end),
        metadata={
            'split_kind': spec.kind.value,
            'validation_horizon': int(validation_horizon),
            'gap': int(spec.gap or 0),
        },
    )


def _build_holdout_folds(batch: ForecastTensorBatch,
                         spec: ForecastingSplitSpec,
                         validation_horizon: int,
                         min_train: int) -> tuple[ForecastingFoldSplit, ...]:
    if batch.series_length <= min_train + validation_horizon:
        raise ValueError('Series is too short for the requested holdout split.')
    test_end = int(batch.series_length)
    test_start = int(test_end - validation_horizon)
    train_start = 0
    train_end = test_start
    return (
        _build_fold_split(
            batch,
            spec=spec,
            fold_index=0,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            validation_horizon=validation_horizon,
        ),
    )


def _build_time_series_split_folds(batch: ForecastTensorBatch,
                                   spec: ForecastingSplitSpec,
                                   validation_horizon: int,
                                   min_train: int) -> tuple[ForecastingFoldSplit, ...]:
    gap = int(max(0, spec.gap))
    remaining = batch.series_length - min_train - gap
    max_possible_splits = remaining // validation_horizon
    n_splits = _resolve_target_fold_count(max_possible_splits, spec.n_splits)
    if n_splits < 1:
        return ()
    initial_test_start = batch.series_length - n_splits * validation_horizon
    folds: list[ForecastingFoldSplit] = []
    for fold_index in range(n_splits):
        test_start = initial_test_start + fold_index * validation_horizon
        test_end = test_start + validation_horizon
        train_end = test_start - gap
        train_start = 0 if spec.max_train_size is None else max(0, train_end - int(spec.max_train_size))
        if train_end - train_start < min_train:
            continue
        folds.append(
            _build_fold_split(
                batch,
                spec=spec,
                fold_index=fold_index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                validation_horizon=validation_horizon,
            )
        )
    return tuple(folds)


def _build_expanding_window_folds(batch: ForecastTensorBatch,
                                  spec: ForecastingSplitSpec,
                                  validation_horizon: int,
                                  min_train: int) -> tuple[ForecastingFoldSplit, ...]:
    gap = int(max(0, spec.gap))
    initial_window = int(spec.initial_window or min_train)
    step_length = int(spec.step_length or validation_horizon)
    folds: list[ForecastingFoldSplit] = []
    fold_index = 0
    train_end = initial_window
    while train_end + gap + validation_horizon <= batch.series_length:
        train_start = 0 if spec.max_train_size is None else max(0, train_end - int(spec.max_train_size))
        if train_end - train_start >= min_train:
            test_start = train_end + gap
            test_end = test_start + validation_horizon
            folds.append(
                _build_fold_split(
                    batch,
                    spec=spec,
                    fold_index=fold_index,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    validation_horizon=validation_horizon,
                )
            )
            fold_index += 1
        train_end += step_length
    target_fold_count = _resolve_target_fold_count(len(folds), spec.n_splits)
    if target_fold_count < 1:
        return ()
    if len(folds) > target_fold_count:
        folds = folds[-target_fold_count:]
    return tuple(folds)


def _build_rolling_window_folds(batch: ForecastTensorBatch,
                                spec: ForecastingSplitSpec,
                                validation_horizon: int,
                                min_train: int) -> tuple[ForecastingFoldSplit, ...]:
    gap = int(max(0, spec.gap))
    window = int(spec.max_train_size or spec.initial_window or min_train)
    step_length = int(spec.step_length or validation_horizon)
    folds: list[ForecastingFoldSplit] = []
    fold_index = 0
    start = 0
    while start + window + gap + validation_horizon <= batch.series_length:
        train_start = start
        train_end = start + window
        test_start = train_end + gap
        test_end = test_start + validation_horizon
        if train_end - train_start >= min_train:
            folds.append(
                _build_fold_split(
                    batch,
                    spec=spec,
                    fold_index=fold_index,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    validation_horizon=validation_horizon,
                )
            )
            fold_index += 1
        start += step_length
    target_fold_count = _resolve_target_fold_count(len(folds), spec.n_splits)
    if target_fold_count < 1:
        return ()
    if len(folds) > target_fold_count:
        folds = folds[-target_fold_count:]
    return tuple(folds)


def iter_forecasting_splits(
        batch: ForecastTensorBatch,
        split_spec: ForecastingSplitSpec | None = None,
) -> tuple[ForecastingFoldSplit, ...]:
    """Build deterministic temporal validation folds for a forecasting batch."""
    spec = split_spec or ForecastingSplitSpec(validation_horizon=batch.forecast_horizon)
    validation_horizon = _resolve_split_horizon(batch, spec)
    min_train = _resolve_min_train_length(batch, spec, validation_horizon)

    if spec.kind in {ForecastingSplitKind.HOLDOUT, ForecastingSplitKind.BLOCKED}:
        folds = _build_holdout_folds(batch, spec, validation_horizon, min_train)
    elif spec.kind == ForecastingSplitKind.TIME_SERIES_SPLIT:
        folds = _build_time_series_split_folds(batch, spec, validation_horizon, min_train)
    elif spec.kind in {ForecastingSplitKind.EXPANDING_WINDOW, ForecastingSplitKind.ROLLING_ORIGIN}:
        folds = _build_expanding_window_folds(batch, spec, validation_horizon, min_train)
    elif spec.kind == ForecastingSplitKind.ROLLING_WINDOW:
        folds = _build_rolling_window_folds(batch, spec, validation_horizon, min_train)
    else:  # pragma: no cover - defensive
        raise ValueError(f'Unsupported forecasting split kind: {spec.kind}')

    if not folds:
        raise ValueError('Series is too short for the requested forecasting split specification.')
    return folds


def split_forecasting_batch(
        batch: ForecastTensorBatch,
        split_spec: ForecastingSplitSpec | None = None,
) -> tuple[ForecastTensorBatch, torch.Tensor]:
    """Return the last train/validation split for compatibility callers."""
    folds = iter_forecasting_splits(batch, split_spec)
    last_fold = folds[-1]
    return last_fold.train_batch, last_fold.validation_target


def evaluate_forecast(
        y_true: Sequence[float] | np.ndarray | torch.Tensor,
        y_pred: Sequence[float] | np.ndarray | torch.Tensor,
        *,
        metric_name: str = 'rmse',
) -> ForecastingEvaluationResult:
    """Evaluate a forecast vector with horizon-wise MAE or RMSE diagnostics."""
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
    """Serialize a ForecastingOperationCapability dataclass."""
    return asdict(capability)

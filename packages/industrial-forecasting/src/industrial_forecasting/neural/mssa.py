"""Standalone MSSAForecaster implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..trajectory import (
    build_page,
    decode_page,
    estimate_window,
    normalize_multivariate_series,
    split_multivariate,
    stack_multivariate,
    truncate_rank,
)
from .runtime import (
    MLPForecastingHead,
    RidgeForecastingHead,
    TensorDevicePolicy,
)
from ..progress import resolve_forecasting_progress_policy


def _ensure_series_length(decoded: np.ndarray, original: np.ndarray) -> np.ndarray:
    if decoded.shape[0] >= original.shape[0]:
        return decoded[:original.shape[0]]
    tail = original[decoded.shape[0]:]
    return np.concatenate([decoded, tail], axis=0)


def _resolve_lag_order(series_length: int, forecast_horizon: int, lag_order: int | None) -> int:
    if lag_order is not None:
        return int(max(1, min(lag_order, series_length - 1)))
    return int(max(1, min(12, forecast_horizon, max(1, series_length // 5))))


@dataclass
class MSSAForecaster:
    """Multichannel SSA forecaster with low-rank denoising and linear/MLP head."""

    forecast_horizon: int
    window_size: int | None = None
    rank: int | None = None
    explained_variance: float = 0.95
    lag_order: int | None = None
    coupled: bool = True
    head_policy: str = 'mlp'
    head_hidden_dim: int = 64
    head_hidden_layers: int = 2
    head_epochs: int = 120
    head_learning_rate: float = 1e-3
    device: str = 'auto'
    dtype: str = 'float32'
    progress_policy: dict[str, object] | bool | None = None
    head_random_seed: int = 42

    def __post_init__(self):
        self.device_policy_ = TensorDevicePolicy(device=self.device, dtype=self.dtype)
        self.progress_policy_ = resolve_forecasting_progress_policy(self.progress_policy)

    def _resolve_head_hidden_dims(self) -> tuple[int, ...]:
        return tuple(int(max(1, self.head_hidden_dim)) for _ in range(int(max(1, self.head_hidden_layers))))

    def _prepare_series(self, time_series: np.ndarray) -> tuple[np.ndarray, int, int, int]:
        normalized = normalize_multivariate_series(time_series)
        series_length, channel_count = normalized.shape
        proposed_window = self.window_size or estimate_window(
            series_length=series_length,
            forecast_horizon=self.forecast_horizon,
            min_ratio=0.10,
            max_ratio=0.25,
        )
        # Page embedding advances by a full window, so the window cannot exceed
        # half of the available history when at least two blocks are required.
        max_window_for_two_blocks = max(2, series_length // 2)
        preferred_minimum = min(
            max(2, self.forecast_horizon + 1),
            max_window_for_two_blocks,
        )
        resolved_window = int(max(
            preferred_minimum,
            min(proposed_window, max_window_for_two_blocks),
        ))
        return normalized, int(series_length), int(channel_count), int(resolved_window)

    def _build_page_results(self, normalized: np.ndarray, resolved_window: int) -> list[object]:
        page_results = [
            build_page(normalized[:, channel_index], window_size=resolved_window)
            for channel_index in range(normalized.shape[1])
        ]
        if any(result.matrix.shape[0] < 2 for result in page_results):
            raise ValueError('mSSA requires at least two Page blocks per channel.')
        return page_results

    def _truncate_page_embeddings(self, page_results: list[object], channel_count: int):
        if self.coupled and channel_count > 1:
            stacked_matrix = stack_multivariate([result.matrix for result in page_results])
            self.embedding_shape_ = tuple(int(value) for value in stacked_matrix.shape)
            truncated = truncate_rank(
                matrix=stacked_matrix,
                rank=self.rank,
                explained_variance=self.explained_variance,
                min_rank=2,
            )
            reconstructed_by_channel = split_multivariate(truncated.reconstructed_matrix, channel_count)
            self.projected_shape_ = tuple(int(value) for value in truncated.projected_states.shape)
            self.basis_shape_ = tuple(int(value) for value in truncated.basis.shape)
            return reconstructed_by_channel, [int(truncated.selected_rank)], [
                float(truncated.explained_variance_retained)]

        per_channel_truncated = [
            truncate_rank(
                matrix=result.matrix,
                rank=self.rank,
                explained_variance=self.explained_variance,
                min_rank=2,
            )
            for result in page_results
        ]
        self.embedding_shape_ = tuple(int(value) for value in page_results[0].matrix.shape)
        self.projected_shape_ = tuple(int(value) for value in per_channel_truncated[0].projected_states.shape)
        self.basis_shape_ = tuple(int(value) for value in per_channel_truncated[0].basis.shape)
        return (
            tuple(item.reconstructed_matrix for item in per_channel_truncated),
            [int(item.selected_rank) for item in per_channel_truncated],
            [float(item.explained_variance_retained) for item in per_channel_truncated],
        )

    def _decode_denoised_series(self,
                                normalized: np.ndarray,
                                page_results: list[object],
                                reconstructed_by_channel) -> np.ndarray:
        denoised_channels = []
        for channel_index, reconstructed in enumerate(reconstructed_by_channel):
            decoded = decode_page(
                reconstructed,
                original_length=page_results[channel_index].diagnostics.original_length,
            )
            denoised_channels.append(_ensure_series_length(decoded, normalized[:, channel_index]))
        return np.column_stack(denoised_channels)

    def _build_head_training_matrices(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lag_order = self.lag_order_
        if series.shape[0] <= lag_order:
            raise ValueError('Series is too short for the selected lag order.')
        design_rows = []
        targets = []
        for index in range(lag_order, series.shape[0]):
            design_rows.append(series[index - lag_order:index].reshape(-1))
            targets.append(series[index])
        return np.asarray(design_rows, dtype=float), np.asarray(targets, dtype=float)

    def _build_forecast_head(self):
        resolved_policy = str(self.head_policy).lower()
        if resolved_policy == 'linear':
            return RidgeForecastingHead(alpha=0.0, device_policy=self.device_policy_)
        return MLPForecastingHead(
            hidden_dims=self._resolve_head_hidden_dims(),
            epochs=self.head_epochs,
            learning_rate=self.head_learning_rate,
            random_seed=self.head_random_seed,
            progress_policy=self.progress_policy_,
            device_policy=self.device_policy_,
        )

    def _fit_forecast_head(self, series: np.ndarray) -> None:
        design, target = self._build_head_training_matrices(series)
        self.head_design_shape_ = tuple(int(value) for value in design.shape)
        self.head_target_shape_ = tuple(int(value) for value in target.shape)
        self.head_policy_ = str(self.head_policy).lower()
        self.head_ = self._build_forecast_head()
        self.head_.fit(design, target)

    def _build_stage_diagnostics(self) -> dict[str, object]:
        decomposition_strategy = 'page_svd_coupled' if self.coupled_ else 'page_svd_per_channel'
        return {
            'trajectory_transform': {
                'kind': 'page',
                'window_size': int(self.window_size_),
                'stride': int(self.page_stride_),
                'forecast_horizon': int(self.forecast_horizon),
                'channel_count': int(self.channel_count_),
                'page_block_count': int(self.page_block_count_),
                'features_shape': tuple(int(value) for value in self.embedding_shape_),
            },
            'decomposition': {
                'strategy': decomposition_strategy,
                'projected_shape': tuple(int(value) for value in self.projected_shape_),
                'basis_shape': tuple(int(value) for value in self.basis_shape_),
                'input_shape': tuple(int(value) for value in self.embedding_shape_),
                'coupled': bool(self.coupled_),
            },
            'rank_truncation': {
                'policy': 'explained_variance',
                'selected_rank': int(self.selected_rank_),
                'explained_variance_retained': float(self.explained_variance_retained_),
                'projected_shape': tuple(int(value) for value in self.projected_shape_),
                'basis_shape': tuple(int(value) for value in self.basis_shape_),
            },
            'forecast_head': {
                'head_type': 'autoregression_head',
                'head_policy': str(self.head_policy_),
                'lag_order': int(self.lag_order_),
                'forecast_horizon': int(self.forecast_horizon),
                'channel_count': int(self.channel_count_),
                'design_shape': tuple(int(value) for value in self.head_design_shape_),
                'target_shape': tuple(int(value) for value in self.head_target_shape_),
                **self.head_.get_diagnostics(),
            },
        }

    def fit(self, time_series: np.ndarray) -> 'MSSAForecaster':
        """Fit Page embedding, rank truncation and autoregressive forecast head."""
        normalized, series_length, channel_count, resolved_window = self._prepare_series(time_series)
        page_results = self._build_page_results(normalized, resolved_window)
        reconstructed_by_channel, selected_ranks, retained_variances = self._truncate_page_embeddings(
            page_results,
            channel_count,
        )
        self.denoised_series_ = self._decode_denoised_series(normalized, page_results, reconstructed_by_channel)
        self.channel_count_ = int(channel_count)
        self.series_length_ = int(series_length)
        self.window_size_ = int(resolved_window)
        self.page_block_count_ = int(page_results[0].matrix.shape[0])
        self.page_stride_ = int(page_results[0].diagnostics.stride)
        self.coupled_ = bool(self.coupled and channel_count > 1)
        self.selected_rank_ = int(max(selected_ranks))
        self.explained_variance_retained_ = float(np.mean(retained_variances))
        self.lag_order_ = _resolve_lag_order(self.series_length_, self.forecast_horizon, self.lag_order)
        self._fit_forecast_head(self.denoised_series_)
        self.diagnostics_ = {
            'model_family': 'low_rank_linear',
            'window_size': self.window_size_,
            'selected_rank': self.selected_rank_,
            'explained_variance_retained': self.explained_variance_retained_,
            'lag_order': self.lag_order_,
            'channel_count': self.channel_count_,
            'page_block_count': self.page_block_count_,
            'coupled': self.coupled_,
            'forecast_horizon': int(self.forecast_horizon),
        }
        self.diagnostics_.update(self._build_stage_diagnostics())
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        """Roll the fitted head forward autoregressively for the requested horizon."""
        horizon = int(forecast_horizon or self.forecast_horizon)
        history = self.denoised_series_ if time_series is None else normalize_multivariate_series(time_series)
        state = history[-self.lag_order_:].copy()
        predictions = []
        for _ in range(horizon):
            next_value = self.head_.predict(state.reshape(1, -1))
            if hasattr(next_value, 'detach'):
                next_value = next_value.detach().cpu().numpy()
            next_value = np.asarray(next_value, dtype=float).reshape(self.channel_count_)
            predictions.append(next_value)
            state = np.vstack([state, next_value])[-self.lag_order_:]
        forecast = np.asarray(predictions, dtype=float)
        return forecast[:, 0] if self.channel_count_ == 1 else forecast

    def get_diagnostics(self) -> dict[str, float | int | bool]:
        """Return stage diagnostics collected during fit."""
        return dict(self.diagnostics_)

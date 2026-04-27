from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd_forecasting import DMDForecaster
from fedot_ind.core.operation.transformation.representation.kernel.kernels import OccupationKernel
from .okhs_common import (
    analyze_okhs_window_size,
    build_okhs_projected_state_sequence,
    build_okhs_trajectory_representation,
    canonical_method_name,
    normalize_okhs_method,
    resolve_okhs_q,
    uses_dmd,
)

try:
    from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
except ImportError:  # pragma: no cover
    class HankelMatrix:
        def __init__(self, time_series, window_size):
            series = np.asarray(time_series)
            self.window_length = window_size
            self.trajectory_matrix = np.array(
                [series[index:index + window_size] for index in range(len(series) - window_size + 1)]
            )

try:
    from fedot.core.data.data import InputData
    from fedot.core.operations.operation_parameters import OperationParameters
    from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
except ImportError:  # pragma: no cover
    InputData = Any
    OperationParameters = dict

    class BaseNeuralModel:
        def __init__(self, params: Optional[OperationParameters] = None):
            self.params = params or {}
            self.epochs = self.params.get('epochs', 100)
            self.learning_rate = self.params.get('learning_rate', 0.001)


class OKHSForecasterTorch(BaseNeuralModel):
    """
    Torch-optimized OKHS forecaster.
    """

    def __init__(self, params: Optional[OperationParameters] = None, **legacy_kwargs):
        merged_params = self._merge_legacy_kwargs(params, legacy_kwargs)
        super().__init__(merged_params)
        self.learning_rate = self.params.get('learning_rate', 0.0001)
        requested_device = self.params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(requested_device)
        self.forecast_horizon = self.params.get('forecast_horizon', 10)
        self.forecasting_strategy = self.params.get('forecasting_strategy', 'multioutput')
        self.n_modes = self.params.get('n_modes', None)
        self.method = normalize_okhs_method(self.params.get('method', 'occupation'))
        self.use_koopman = self.params.get('use_koopman', True)
        self.q = self.params.get('q', 0.7)
        self.q_policy = self.params.get('q_policy', 'fixed')
        self.q_selector = self.params.get('q_selector', None)
        self.window_policy = self.params.get('window_policy', 'fixed')
        self.trajectory_sampling_policy = self.params.get('trajectory_sampling_policy', 'adaptive_stride')
        self.trajectory_rank_policy = self.params.get('trajectory_rank_policy', 'explained_dispersion')
        self.trajectory_rank_value = self.params.get('trajectory_rank_value', None)
        self.trajectory_representation_policy = self.params.get('trajectory_representation_policy', 'projected')
        self.latent_trajectory_stride_policy = self.params.get('latent_trajectory_stride_policy', 'adaptive')
        self.latent_trajectory_stride = self.params.get('latent_trajectory_stride', None)
        self.kernel_type = self.params.get('kernel_type', 'rbf')
        self.kernel_ = None
        self.resolved_q_ = self.q
        self.resolved_window_size_ = None
        self.window_diagnostics_ = None
        self.trajectory_preprocessing_ = None
        self.projection_metadata_ = None
        self._projection_runtime_ = None
        self.method_name_ = canonical_method_name(self.method)

    @staticmethod
    def _merge_legacy_kwargs(params, legacy_kwargs):
        merged = dict(params or {})
        merged.update(legacy_kwargs)
        if 'horizon' in merged and 'forecast_horizon' not in merged:
            merged['forecast_horizon'] = merged.pop('horizon')
        if 'max_epochs' in merged and 'epochs' not in merged:
            merged['epochs'] = merged.pop('max_epochs')
        return merged

    def _init_decomposition_strategy(self):
        if uses_dmd(self.method):
            self.dmd_model = DMDForecaster(
                forecast_horizon=self.forecast_horizon,
                n_modes=self.n_modes,
                use_koopman=self.use_koopman,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                device=self.device,
            )
        else:
            self.kernel_ = OccupationKernel(q=self.resolved_q_, kernel_type=self.kernel_type)

    def _hankelize(self, time_series, window_size):
        self.hankel_matrix = HankelMatrix(time_series=time_series, window_size=window_size)

    def fit(self, time_series, window_size=20):
        self.window_diagnostics_ = analyze_okhs_window_size(
            window_size=window_size,
            window_policy=self.window_policy,
            time_series=time_series,
            forecast_horizon=self.forecast_horizon,
        )
        self.resolved_window_size_ = self.window_diagnostics_["resolved_window_size"]
        self._hankelize(time_series, self.resolved_window_size_)
        self.projection_metadata_ = None
        self._projection_runtime_ = None

        if uses_dmd(self.method):
            representation = build_okhs_trajectory_representation(
                time_series=time_series,
                window_size=self.resolved_window_size_,
                window_policy=self.window_policy,
                forecast_horizon=self.forecast_horizon,
                trajectory_sampling_policy=self.trajectory_sampling_policy,
                trajectory_rank_policy=self.trajectory_rank_policy,
                trajectory_rank_value=self.trajectory_rank_value,
                trajectory_representation_policy=self.trajectory_representation_policy,
                latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
                latent_trajectory_stride=self.latent_trajectory_stride,
            )
            training_matrix = np.asarray(representation["training_matrix"], dtype=float)
            self.trajectory_preprocessing_ = representation["trajectory_preprocessing"]
            self.projection_metadata_ = representation["projection_metadata"]
            self._projection_runtime_ = representation["projection_runtime"]
            self.hankel_matrix.trajectory_matrix = training_matrix
            if hasattr(self.hankel_matrix, "window_length"):
                self.hankel_matrix.window_length = self.resolved_window_size_
        else:
            self.trajectory_preprocessing_ = None

        trajectories_for_q = self.hankel_matrix.trajectory_matrix
        self.resolved_q_ = resolve_okhs_q(
            q=self.q,
            q_policy=self.q_policy,
            trajectories=trajectories_for_q,
            q_selector=self.q_selector,
        )
        self._init_decomposition_strategy()
        if uses_dmd(self.method):
            return self.dmd_model.fit(self.hankel_matrix.trajectory_matrix, self.resolved_window_size_)
        return self._fit_direct_okhs_torch()

    def _train_loop(self, val_loader, loss_fn, optimizer, train_loader=None):
        del train_loader
        optimizer.zero_grad()
        predictions = self.gram_matrix @ self.weights_
        mse_loss = loss_fn(predictions, val_loader)
        reg_loss = torch.norm(self.weights_, p=2) * 0.01
        total_loss = mse_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        return total_loss

    def _fit_direct_okhs_torch(self):
        x_trajectories, y_targets = self._from_trajectory_to_features(self.hankel_matrix.trajectory_matrix.T)
        self.train_traj, self.targets = x_trajectories[:-1, :], y_targets[:-1, :]
        weight_shape = (
            len(self.train_traj), self.forecast_horizon) if self.forecasting_strategy == 'multioutput' else len(
            self.train_traj)
        self.weights_ = nn.Parameter(torch.randn(weight_shape, device=self.device) * 0.01)
        self.gram_matrix = torch.tensor(
            self.kernel_.compute_gram_matrix(self.train_traj),
            dtype=torch.float32,
            device=self.device,
        )
        optimizer = torch.optim.Adam([self.weights_], lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            self._train_loop(optimizer=optimizer, loss_fn=loss_fn, val_loader=self.targets)
        return self

    def _uses_projected_representation(self):
        return (
            self.projection_metadata_ is not None
            and self.projection_metadata_.get('representation_policy') == 'projected'
            and self.projection_metadata_.get('decode_supported') is True
            and self._projection_runtime_ is not None
        )

    def _resolve_projected_initial_trajectory(self, input_data: InputData = None):
        runtime = self._projection_runtime_ or {}
        latent_window_size = int(runtime.get("latent_window_size", 0))
        if latent_window_size <= 0:
            raise ValueError("Projected OKHS DMD path requires latent_window_size.")

        if input_data is None:
            latent_states = np.asarray(runtime["latent_state_matrix"], dtype=float)
            sampled_matrix = np.asarray(runtime["sampled_matrix"], dtype=float)
            return latent_states[-latent_window_size:], sampled_matrix[-latent_window_size:]

        sampled_matrix, latent_states = build_okhs_projected_state_sequence(
            time_series=input_data.features,
            window_size=self.resolved_window_size_,
            effective_stride=int(self.trajectory_preprocessing_["effective_stride"]),
            basis=np.asarray(runtime["basis"], dtype=float),
        )
        return latent_states[-latent_window_size:], sampled_matrix[-latent_window_size:]

    def predict(self, input_data: InputData = None, output_mode: str = 'default') -> np.array:
        del output_mode
        if uses_dmd(self.method):
            if self._uses_projected_representation():
                latent_trajectory, _ = self._resolve_projected_initial_trajectory(input_data)
                latent_prediction = np.asarray(self.dmd_model.predict(latent_trajectory), dtype=float)
                if latent_prediction.ndim == 1:
                    latent_prediction = latent_prediction.reshape(self.forecast_horizon, -1)
                basis = np.asarray(self._projection_runtime_["basis"], dtype=float)
                decoded_prediction = latent_prediction @ basis.T
                return decoded_prediction[:, -1].reshape(-1)

            last_trajectory = input_data.features[-self.hankel_matrix.window_length:] if input_data is not None \
                else self.hankel_matrix.trajectory_matrix[-1]
            return self.dmd_model.predict(last_trajectory)

        last_trajectory = input_data.features[-self.hankel_matrix.window_length:] if input_data is not None \
            else self.hankel_matrix.trajectory_matrix.T[-1]
        return self._predict_direct_torch(last_trajectory)

    def _predict_direct_torch(self, last_trajectory):
        predictions = []
        current_trajectory = last_trajectory.copy()
        current_trajectory = current_trajectory[-self.forecast_horizon:]

        def _prediction_loop(current_window):
            kernels = [self.kernel_._compute_trajectory_kernel(current_window, train_traj) for train_traj in
                       self.train_traj]
            kernels_tensor = torch.tensor(kernels, dtype=torch.float32, device=self.device)
            prediction = kernels_tensor @ self.weights_
            if self.forecasting_strategy != 'multioutput':
                current_window = np.roll(current_window, -1)
                current_window[-1] = prediction.cpu().numpy()
            return prediction.cpu().numpy(), current_window

        with torch.no_grad():
            if self.forecasting_strategy == 'multioutput':
                predictions, current_trajectory = _prediction_loop(current_trajectory)
            else:
                for _ in range(self.forecast_horizon):
                    prediction, current_trajectory = _prediction_loop(current_trajectory)
                    predictions.append(prediction)

        return np.array(predictions).flatten()

    def get_optimization_info(self):
        info = {
            'method': self.method_name_,
            'device': str(self.device),
            'q': self.resolved_q_,
            'q_policy': self.q_policy,
            'forecast_horizon': self.forecast_horizon,
            'window_policy': self.window_policy,
            'trajectory_sampling_policy': self.trajectory_sampling_policy,
            'trajectory_rank_policy': self.trajectory_rank_policy,
            'trajectory_rank_value': self.trajectory_rank_value,
            'trajectory_representation_policy': self.trajectory_representation_policy,
            'latent_trajectory_stride_policy': self.latent_trajectory_stride_policy,
            'latent_trajectory_stride': self.latent_trajectory_stride,
            'resolved_window_size': self.resolved_window_size_,
            'window_diagnostics': self.window_diagnostics_,
            'trajectory_preprocessing': self.trajectory_preprocessing_,
            'projection_metadata': self.projection_metadata_,
        }
        if hasattr(self, 'weights_'):
            info['weights_norm'] = torch.norm(self.weights_).item()
        return info

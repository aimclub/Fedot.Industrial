import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
from fedot_ind.core.models.kernel.okhs_runtime import build_okhs_fit_plan

from fedot_ind.core.models.ts_forecasting.dmd_models.deep_encoder_training import (
    DeepEncoderTrainingConfig,
    train_deep_encoder
)

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fdmd_net import DeepFDMDAutoencoder
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.kernels import DeepKernel
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.gram_transform import OKHSTransformer
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.fractional_liouville import FractionalLiouvilleOperator
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.fractional_dmd import FractionalDMD


class InferenceEncoderAdapter(nn.Module):
    """
    Адаптер для DeepKernel.
    Конвертирует данные в тензоры и отключает градиенты во время аналитической Фазы 2.
    """
    def __init__(self, autoencoder: DeepFDMDAutoencoder, device: str):
        super().__init__()
        self.autoencoder = autoencoder
        self.device = device

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_t = torch.tensor(x, dtype=torch.float64, device=self.device)
            else:
                x_t = x.to(dtype=torch.float64, device=self.device)

            orig_shape = x_t.shape
            if x_t.ndim > 2:
                x_flat = x_t.view(-1, orig_shape[-1])
                z_flat = self.autoencoder.encode_trajectory(x_flat)
                return z_flat.view(*orig_shape[:-1], -1)
            return self.autoencoder.encode_trajectory(x_t)


class DeepOKHSForecasterTorch(OKHSForecaster):
    """
    Decoupled Spectral Training:
    Фаза 1: Делегируется внешнему процессу train_deep_encoder (или используются предобученные веса).
    Фаза 2: Спектральное разложение (FractionalDMD) с использованием DeepKernel.
    """
    def __init__(self, params: Optional[OperationParameters] = None, **kwargs):
        merged_params = dict(params or {})
        merged_params.update(kwargs)
        
        self.encoder_state_dict = merged_params.pop('encoder_state_dict', None)
        
        self.latent_dim = merged_params.pop('latent_dim', 16)
        self.ae_epochs = merged_params.pop('ae_epochs', 100)
        self.ae_learning_rate = merged_params.pop('ae_learning_rate', 1e-3)
        self.alpha_adjoint = merged_params.pop('alpha_adjoint', 1.0)
        self.beta_rec = merged_params.pop('beta_rec', 1.0)
        self.hidden_layers = merged_params.pop('hidden_layers', [64, 64])
        self.dt = merged_params.pop('dt', 1.0)
        self.device = merged_params.get('device', 'cpu')
        
        super().__init__(**merged_params)

    def fit(self, time_series: np.ndarray, window_size: int = 20):
        fit_plan = build_okhs_fit_plan(
            time_series=time_series,
            window_size=window_size,
            method=self.method,
            forecast_horizon=self.forecast_horizon,
            q=self.q,
            q_policy=self.q_policy,
            q_selector=self.q_selector,
            window_policy=self.window_policy,
            trajectory_sampling_policy=self.trajectory_sampling_policy,
            trajectory_rank_policy=self.trajectory_rank_policy,
            trajectory_rank_value=self.trajectory_rank_value,
            trajectory_representation_policy=self.trajectory_representation_policy,
            latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
            latent_trajectory_stride=self.latent_trajectory_stride,
        )
        self.train_series_ = fit_plan['train_series']
        self.resolved_window_size_ = fit_plan['resolved_window_size']
        self.window_size_ = self.resolved_window_size_
        self.window_diagnostics_ = fit_plan['window_diagnostics']

        self.trajectories_ = fit_plan['trajectories'] 
        self.trajectory_preprocessing_ = fit_plan['trajectory_preprocessing']
        self.projection_metadata_ = fit_plan['projection_metadata']
        self._projection_runtime_ = fit_plan['projection_runtime']
        self.resolved_q_ = fit_plan['resolved_q']
        
        d_features = 1 if self.trajectories_.ndim == 2 else self.trajectories_.shape[-1]

        # ФАЗА 1: Обучение Автоэнкодера
        if self.encoder_state_dict is None:
            training_config = DeepEncoderTrainingConfig(
                latent_dim=self.latent_dim,
                epochs=self.ae_epochs,
                learning_rate=self.ae_learning_rate,
                alpha_adjoint=self.alpha_adjoint,
                reconstruction_weight=self.beta_rec,
                hidden_layers=self.hidden_layers,
                dt=self.dt,
                q=self.resolved_q_,
                device=self.device
            )
            training_result = train_deep_encoder(
                trajectories=self.trajectories_, 
                config=training_config
            )
            self.encoder_state_dict = training_result.encoder_state_dict

        self.autoencoder = DeepFDMDAutoencoder(
            input_dim=d_features,
            latent_dim=self.latent_dim,
            hidden_layers=self.hidden_layers,
            dtype=torch.float64
        ).to(self.device)
        self.autoencoder.load_state_dict(self.encoder_state_dict)
        
        # ФАЗА 2: Аналитическое разложение через DeepKernel
        self.autoencoder.eval()

        encoder_adapter = InferenceEncoderAdapter(self.autoencoder, self.device)
        self.kernel_ = DeepKernel(feature_extractor=encoder_adapter, base_kernel=None)

        self.okhs_transformer = OKHSTransformer(
            kernel=self.kernel_,
            q=self.resolved_q_,
            dt=self.dt,
            device=self.device
        )
        
        self.liouville_op = FractionalLiouvilleOperator(
            okhs_transformer=self.okhs_transformer,
            verbose=False
        )

        self.model = FractionalDMD(
            liouville_operator=self.liouville_op,
            device=self.device
        )

        self.okhs_transformer.fit(self.trajectories_)
        self.liouville_op.fit(self.trajectories_)
        self.model.fit(self.trajectories_)
        
        return self
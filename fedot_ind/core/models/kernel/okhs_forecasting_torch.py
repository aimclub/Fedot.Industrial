from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd_forecasting import DMDForecaster
from fedot_ind.core.operation.transformation.representation.kernel.kernels import OccupationKernel
from .okhs_common import analyze_okhs_window_size, canonical_method_name, normalize_okhs_method, resolve_okhs_q, \
    uses_dmd

try:
    from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
except ImportError:  # pragma: no cover - lightweight fallback for local tests without extra deps
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
except ImportError:  # pragma: no cover - lightweight fallback for local tests without fedot
    InputData = Any
    OperationParameters = dict


    class BaseNeuralModel:
        def __init__(self, params: Optional[OperationParameters] = None):
            self.params = params or {}
            self.epochs = self.params.get('epochs', 100)
            self.learning_rate = self.params.get('learning_rate', 0.001)


class OKHSForecasterTorch(BaseNeuralModel):
    """
    Прогнозирование временных рядов с использованием Occupation Kernel Hilbert Spaces
    и дробных операторов Лиувилля с оптимизацией через PyTorch
    """

    def __init__(self, params: Optional[OperationParameters] = None, **legacy_kwargs):
        merged_params = self._merge_legacy_kwargs(params, legacy_kwargs)
        super().__init__(merged_params)
        # learning params
        self.learning_rate = self.params.get('learning_rate', 0.0001)
        requested_device = self.params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(requested_device)
        # forecasting params
        self.forecast_horizon = self.params.get('forecast_horizon', 10)
        self.forecasting_strategy = self.params.get('forecasting_strategy', 'multioutput')
        # dmd params
        self.n_modes = self.params.get('n_modes', None)
        self.method = normalize_okhs_method(self.params.get('method', 'occupation'))
        self.use_koopman = self.params.get('use_koopman', True)
        # kernel params
        self.q = self.params.get('q', 0.7)
        self.q_policy = self.params.get('q_policy', 'fixed')
        self.q_selector = self.params.get('q_selector', None)
        self.window_policy = self.params.get('window_policy', 'fixed')
        self.kernel_type = self.params.get('kernel_type', 'rbf')
        self.kernel_ = None
        self.resolved_q_ = self.q
        self.resolved_window_size_ = None
        self.window_diagnostics_ = None
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
            self.dmd_model = DMDForecaster(forecast_horizon=self.forecast_horizon,
                                           n_modes=self.n_modes, use_koopman=self.use_koopman,
                                           learning_rate=self.learning_rate, epochs=self.epochs,
                                           device=self.device)
        else:
            self.kernel_ = OccupationKernel(q=self.resolved_q_, kernel_type=self.kernel_type)

    def _hankelize(self, time_series, window_size):
        """Создание траекторий из временного ряда"""
        self.hankel_matrix = HankelMatrix(time_series=time_series, window_size=window_size)

    def fit(self, time_series, window_size=20):
        """Обучение модели на временном ряде"""
        self.window_diagnostics_ = analyze_okhs_window_size(
            window_size=window_size,
            window_policy=self.window_policy,
            time_series=time_series,
            forecast_horizon=self.forecast_horizon,
        )
        self.resolved_window_size_ = self.window_diagnostics_["resolved_window_size"]
        self._hankelize(time_series, self.resolved_window_size_)
        self.resolved_q_ = resolve_okhs_q(
            q=self.q,
            q_policy=self.q_policy,
            trajectories=self.hankel_matrix.trajectory_matrix.T,
            q_selector=self.q_selector,
        )
        self._init_decomposition_strategy()
        return self.dmd_model.fit(self.hankel_matrix.trajectory_matrix.T,
                                  self.resolved_window_size_) if uses_dmd(
            self.method) else self._fit_direct_okhs_torch()

    def _train_loop(self, val_loader, loss_fn, optimizer, train_loader=None):
        optimizer.zero_grad()
        # Предсказания
        predictions = self.gram_matrix @ self.weights_
        # Loss с L2 регуляризацией
        mse_loss = loss_fn(predictions, val_loader)
        reg_loss = torch.norm(self.weights_, p=2) * 0.01
        total_loss = mse_loss + reg_loss

        total_loss.backward()
        optimizer.step()
        return total_loss

    def _fit_direct_okhs_torch(self):
        """Прямое обучение в OKHS через PyTorch"""
        X_trajectories, y_targets = self._from_trajectory_to_features(self.hankel_matrix.trajectory_matrix.T)
        self.train_traj, self.targets = X_trajectories[:-1, :], y_targets[:-1,
                                                                :]  # убираем последнюю траекторию из обучаемых весов
        # Веса модели
        if self.forecasting_strategy == 'multioutput':
            weight_shape = (len(self.train_traj), self.forecast_horizon)
        else:
            weight_shape = len(self.train_traj)

        self.weights_ = nn.Parameter(torch.randn(weight_shape, device=self.device) * 0.01)
        # Матрица Грама для обучения
        self.gram_matrix = torch.tensor(self.kernel_.compute_gram_matrix(self.train_traj),
                                        dtype=torch.float32, device=self.device)
        # Оптимизатор
        optimizer = torch.optim.Adam([self.weights_], lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        for epoch in range(self.epochs):
            total_loss = self._train_loop(optimizer=optimizer, loss_fn=loss_fn,
                                          val_loader=self.targets)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {total_loss.item():.4f}')
        return self

    def predict(self, input_data: InputData = None, output_mode: str = 'default') -> np.array:
        """Прогнозирование будущих значений"""
        last_trajectory = input_data.features[-self.hankel_matrix.window_length:] if input_data is not None \
            else self.hankel_matrix.trajectory_matrix.T[-1]

        return self.dmd_model.predict(last_trajectory) if uses_dmd(self.method) \
            else self._predict_direct_torch(last_trajectory)

    def _predict_direct_torch(self, last_trajectory):
        """Прямой прогноз с использованием OKHS"""
        predictions = []
        current_trajectory = last_trajectory.copy()
        current_trajectory = current_trajectory[-self.forecast_horizon:]

        def _prediction_loop(current_trajectory):
            kernels = [self.kernel_._compute_trajectory_kernel(current_trajectory, train_traj)
                       for train_traj in self.train_traj]
            kernels_tensor = torch.tensor(kernels, dtype=torch.float32, device=self.device)

            # Предсказание как линейная комбинация
            prediction = kernels_tensor @ self.weights_
            if self.forecasting_strategy != 'multioutput':
                # Обновляем траекторию для следующего шага
                current_trajectory = np.roll(current_trajectory, -1)
                current_trajectory[-1] = prediction.cpu().numpy()
            return prediction.cpu().numpy(), current_trajectory

        with torch.no_grad():
            if self.forecasting_strategy == 'multioutput':
                predictions, current_trajectory = _prediction_loop(current_trajectory)
            else:
                for step in range(self.forecast_horizon):
                    prediction, current_trajectory = _prediction_loop(current_trajectory)
                    predictions.append(prediction)

        return np.array(predictions).flatten()

    def get_optimization_info(self):
        """Информация об оптимизации"""
        info = {'method': self.method_name_, 'device': str(self.device), 'q': self.resolved_q_,
                'q_policy': self.q_policy,
                'forecast_horizon': self.forecast_horizon,
                'window_policy': self.window_policy,
                'resolved_window_size': self.resolved_window_size_,
                'window_diagnostics': self.window_diagnostics_}
        if hasattr(self, 'eigenvalues_'):
            info['eigenvalues'] = self.eigenvalues_
        if hasattr(self, 'weights_'):
            info['weights_norm'] = torch.norm(self.weights_).item()
        return info

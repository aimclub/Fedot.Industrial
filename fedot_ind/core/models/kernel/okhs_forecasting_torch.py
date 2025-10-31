from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd_forecasting import DMDForecaster
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.representation.kernel.kernels import OccupationKernel
from fedot_ind.core.operation.transformation.representation.kernel.utils import mittag_leffler


class OKHSForecasterTorch(BaseNeuralModel):
    """
    Прогнозирование временных рядов с использованием Occupation Kernel Hilbert Spaces
    и дробных операторов Лиувилля с оптимизацией через PyTorch
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        # learning params
        self.learning_rate = self.params.get('learning_rate', 0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # forecasting params
        self.forecast_horizon = self.params.get('forecast_horizon', 10)
        self.forecasting_strategy = self.params.get('forecasting_strategy', 'multioutput')
        # dmd params
        self.n_modes = self.params.get('n_modes', None)
        self.method = self.params.get('method', 'dmd')
        self.use_koopman = self.params.get('use_koopman', True)
        # kernel params
        self.q = self.params.get('q', 0.7)
        self.kernel_ = None

    def _init_decomposition_strategy(self):
        if self.method == 'dmd':
            self.dmd_model = DMDForecaster(forecast_horizon=self.forecast_horizon,
                                           n_modes=self.n_modes, use_koopman=self.use_koopman,
                                           learning_rate=self.learning_rate, epochs=self.epochs,
                                           device=self.device)
        else:
            self.kernel_ = OccupationKernel(q=self.q)

    def _hankelize(self, time_series, window_size):
        """Создание траекторий из временного ряда"""
        self.hankel_matrix = HankelMatrix(time_series=time_series, window_size=window_size)

    def _get_feature_target(self):
        """
        Создание траекторий с многомерными целевыми переменными
        """
        # Создаем пары (фичи это лаги за предыдущие периоды, таргет-значение в текущий момент времени)
        X_trajectories = self.hankel_matrix.trajectory_matrix.T[:-1]
        if self.forecasting_strategy == 'multioutput':
            y_targets = [traj[:self.forecast_horizon] for traj in self.hankel_matrix.trajectory_matrix.T[1:]]
        else:
            y_targets = [traj[-1] for traj in self.hankel_matrix.trajectory_matrix.T[1:]]
        y_targets = torch.tensor(y_targets, dtype=torch.float32, device=self.device)

        return X_trajectories, y_targets

    def fit(self, time_series, window_size=20):
        """Обучение модели на временном ряде"""
        self._hankelize(time_series, window_size)
        self._init_decomposition_strategy()
        return self.dmd_model.fit(self.hankel_matrix.trajectory_matrix.T,
                                  window_size) if self.method == 'dmd' else self._fit_direct_okhs_torch()

    def _fit_dmd_torch(self, window_size):
        """Обучение DMD модели через PyTorch"""
        # Преобразуем траектории в тензоры
        trajectories_tensor = torch.tensor(self.hankel_matrix.trajectory_matrix,
                                           dtype=torch.float32, device=self.device)

        return self.dmd_model.fit(trajectories_tensor, window_size)

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
        X_trajectories, y_targets = self._get_feature_target()
        # Веса модели
        if self.forecasting_strategy == 'multioutput':
            weight_shape = (len(X_trajectories), self.forecast_horizon)
        else:
            weight_shape = len(X_trajectories)

        self.weights_ = nn.Parameter(torch.randn(weight_shape, device=self.device) * 0.01)
        # Матрица Грама для обучения
        self.gram_matrix = torch.tensor(self.kernel_.compute_gram_matrix(X_trajectories),
                                        dtype=torch.float32, device=self.device)
        # Оптимизатор
        optimizer = torch.optim.Adam([self.weights_], lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        for epoch in range(self.epochs):
            total_loss = self._train_loop(optimizer=optimizer, loss_fn=loss_fn,
                                          val_loader=y_targets)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {total_loss.item():.4f}')
        return self

    def predict(self, input_data: InputData = None, output_mode: str = 'default') -> np.array:
        """Прогнозирование будущих значений"""
        last_trajectory = input_data.features[-self.hankel_matrix.window_length:] if input_data is not None \
            else self.hankel_matrix.trajectory_matrix.T[-1]

        return self.dmd_model.predict(last_trajectory) if self.method == 'dmd' \
            else self._predict_direct_torch(last_trajectory)

    def _predict_dmd_torch(self, last_trajectory):
        """DMD прогноз с использованием функции Миттаг-Леффлера"""
        predictions = []
        current_state = torch.tensor(last_trajectory, dtype=torch.float32, device=self.device)

        for t in range(1, self.forecast_horizon + 1):
            state_pred = torch.zeros_like(current_state)

            # Используем функцию Миттаг-Леффлера для дробной динамики
            for i, eigenvalue in enumerate(self.eigenvalues_):
                ml_value = mittag_leffler(eigenvalue * t ** self.q, self.q)
                # Упрощенная проекция на собственные векторы
                contribution = current_state[i] * ml_value if i < len(current_state) else 0
                state_pred[i % len(current_state)] += contribution.real

            predictions.append(state_pred.cpu().numpy().copy())
            current_state = state_pred

        return np.array(predictions).flatten()

    def _predict_direct_torch(self, last_trajectory):
        """Прямой прогноз с использованием OKHS"""
        predictions = []
        current_trajectory = last_trajectory.copy()

        def _prediction_loop(current_trajectory):
            kernels = [self.kernel_._compute_trajectory_kernel(current_trajectory, train_traj)
                       for train_traj in self.hankel_matrix.trajectory_matrix.T[:-1]]
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
        info = {
            'method': self.method,
            'device': str(self.device),
            'q': self.q,
            'forecast_horizon': self.forecast_horizon
        }

        if hasattr(self, 'eigenvalues_'):
            info['eigenvalues'] = self.eigenvalues_
        if hasattr(self, 'weights_'):
            info['weights_norm'] = torch.norm(self.weights_).item()

        return info

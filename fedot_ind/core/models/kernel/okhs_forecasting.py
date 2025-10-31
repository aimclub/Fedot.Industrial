import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd import FractionalDMD
from ...operation.transformation.representation.kernel.kernels import OccupationKernel


class OKHSForecaster(BaseEstimator, RegressorMixin):
    """
    Прогнозирование временных рядов с использованием Occupation Kernel Hilbert Spaces
    и дробных операторов Лиувилля
    """

    def __init__(self, q=0.7, forecast_horizon=10, n_modes=5, method='dmd'):
        self.q = q
        self.forecast_horizon = forecast_horizon
        self.n_modes = n_modes
        self.method = method

        if method == 'dmd':
            self.model = FractionalDMD(q=q, n_modes=n_modes)
        else:
            self.model = None

    def _create_trajectories(self, time_series, window_size):
        """Создание траекторий из временного ряда"""
        trajectories = []
        for i in range(len(time_series) - window_size):
            trajectory = time_series[i:i + window_size]
            trajectories.append(trajectory)
        return trajectories

    def fit(self, time_series, window_size=20):
        """Обучение модели на временном ряде"""
        self.window_size_ = window_size
        self.trajectories_ = self._create_trajectories(time_series, window_size)

        if self.method == 'dmd':
            self.model.fit(self.trajectories_)
        elif self.method == 'direct':
            self._fit_direct_okhs(time_series)

        return self

    def _fit_direct_okhs(self, time_series):
        """Прямое обучение в OKHS без DMD"""
        self.kernel_ = OccupationKernel(q=self.q)

        # Создаем пары (вход, цель) для регрессии
        X_trajectories = self.trajectories_[:-1]
        y_targets = [traj[-1] for traj in self.trajectories_[1:]]

        # Матрица Грама для обучения
        self.gram_matrix_ = self.kernel_.compute_gram_matrix(X_trajectories)

        # Решаем задачу регрессии
        self.weights_ = np.linalg.lstsq(
            self.gram_matrix_, y_targets, rcond=None
        )[0]

    def predict(self, time_series=None):
        """Прогнозирование будущих значений"""
        if time_series is None:
            # Продолжение обученного ряда
            last_trajectory = self.trajectories_[-1]
        else:
            # Прогноз для нового ряда
            last_trajectory = time_series[-self.window_size_:]

        if self.method == 'dmd':
            future_times = np.arange(1, self.forecast_horizon + 1)
            predictions = self.model.predict(last_trajectory, future_times)
            return predictions.flatten()
        else:
            return self._predict_direct(last_trajectory)

    def _predict_direct(self, last_trajectory):
        """Прямой прогноз с использованием OKHS"""
        predictions = []
        current_trajectory = last_trajectory.copy()

        for _ in range(self.forecast_horizon):
            # Вычисляем ядра с обучающими траекториями
            kernels = []
            for train_traj in self.trajectories_[:-1]:
                kernel_val = self.kernel_._compute_trajectory_kernel(
                    current_trajectory, train_traj
                )
                kernels.append(kernel_val)

            kernels = np.array(kernels)

            # Предсказание как линейная комбинация
            prediction = kernels @ self.weights_
            predictions.append(prediction)

            # Обновляем траекторию для следующего шага
            current_trajectory = np.roll(current_trajectory, -1)
            current_trajectory[-1] = prediction

        return np.array(predictions)

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd import FractionalDMD
from ...operation.transformation.representation.kernel.kernels import OccupationKernel
from .okhs_common import (
    analyze_okhs_window_size,
    canonical_method_name,
    normalize_okhs_method,
    resolve_okhs_q,
    uses_dmd,
)


class OKHSForecaster(BaseEstimator, RegressorMixin):
    """
    Прогнозирование временных рядов с использованием Occupation Kernel Hilbert Spaces
    и дробных операторов Лиувилля
    """

    def __init__(
            self,
            q=0.7,
            forecast_horizon=10,
            n_modes=5,
            method='dmd',
            forecasting_strategy='recursive',
            q_policy='fixed',
            q_selector=None,
            window_policy='adaptive_cycle_aware',
    ):
        self.q = q
        self.forecast_horizon = forecast_horizon
        self.n_modes = n_modes
        self.method = normalize_okhs_method(method)
        self.forecasting_strategy = forecasting_strategy
        self.q_policy = q_policy
        self.q_selector = q_selector
        self.window_policy = window_policy
        self.model = None
        self.resolved_q_ = q
        self.resolved_window_size_ = None
        self.window_diagnostics_ = None
        self.method_name_ = canonical_method_name(self.method)

    def _create_trajectories(self, time_series, window_size):
        """Создание траекторий из временного ряда"""
        trajectories = []
        for i in range(len(time_series) - window_size):
            trajectory = time_series[i:i + window_size]
            trajectories.append(trajectory)
        return trajectories

    def fit(self, time_series, window_size=20):
        """Обучение модели на временном ряде"""
        self.window_diagnostics_ = analyze_okhs_window_size(
            window_size=window_size,
            window_policy=self.window_policy,
            time_series=time_series,
            forecast_horizon=self.forecast_horizon,
        )
        self.resolved_window_size_ = self.window_diagnostics_["resolved_window_size"]
        self.window_size_ = self.resolved_window_size_
        self.trajectories_ = self._create_trajectories(time_series, self.window_size_)
        self.resolved_q_ = resolve_okhs_q(
            q=self.q,
            q_policy=self.q_policy,
            trajectories=self.trajectories_,
            q_selector=self.q_selector,
        )

        if uses_dmd(self.method):
            self.model = FractionalDMD(q=self.resolved_q_, n_modes=self.n_modes)
            self.model.fit(self.trajectories_)
        else:
            self._fit_direct_okhs(time_series)

        return self

    def _fit_direct_okhs(self, time_series):
        """Прямое обучение в OKHS без DMD"""
        self.kernel_ = OccupationKernel(q=self.resolved_q_)

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

        if uses_dmd(self.method):
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

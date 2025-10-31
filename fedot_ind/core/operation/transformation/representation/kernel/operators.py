import numpy as np
from scipy.special import gamma

from .base import FractionalBase


class FractionalLiouvilleOperator(FractionalBase):
    """Дробный оператор Лиувилля"""

    def __init__(self, q=0.7):
        super().__init__(q)

    def apply(self, function, trajectory, current_time):
        """Применение оператора к функции вдоль траектории"""
        result = 0.0

        for i, time_val in enumerate(trajectory.times):
            if time_val < current_time:
                # Ядро памяти
                memory_weight = (current_time - time_val) ** (-self.q) / gamma(1 - self.q)

                # Градиент функции (упрощенная версия)
                grad_f = self._numerical_gradient(function, trajectory.values[i])

                # Вклад в результат
                result += grad_f * memory_weight

        return result

    def _numerical_gradient(self, func, point, eps=1e-6):
        """Численное вычисление градиента"""
        if isinstance(point, (int, float)):
            return (func(point + eps) - func(point - eps)) / (2 * eps)
        else:
            grad = np.zeros_like(point)
            for i in range(len(point)):
                point_plus = point.copy()
                point_minus = point.copy()
                point_plus[i] += eps
                point_minus[i] -= eps
                grad[i] = (func(point_plus) - func(point_minus)) / (2 * eps)
            return grad


class MemoryWeightComputer(FractionalBase):
    """Вычисление весов памяти для временных рядов"""

    def compute_weights(self, time_series):
        """Вычисление весов для каждого временного шага"""
        n = len(time_series)
        weights = np.zeros(n)

        for i in range(n):
            # Вес убывает как степенная функция
            time_from_end = n - i
            weights[i] = time_from_end ** (-self.q) / gamma(1 - self.q)

        # Нормализация
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights

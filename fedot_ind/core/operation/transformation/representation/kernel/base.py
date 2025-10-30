import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma, kv

from fedot_ind.core.operation.transformation.representation.kernel.utils import mittag_leffler


class KernelBase(nn.Module):
    """Базовый класс для ядерных функций с расширенным функционалом"""

    def __init__(self, **kernel_params):
        super().__init__()
        self.kernel_params = kernel_params

    def forward(self, X, Y=None):
        """Основной метод для вычисления ядерной матрицы"""
        if Y is None:
            Y = X

        n_x = X.shape[0] if hasattr(X, 'shape') else len(X)
        n_y = Y.shape[0] if hasattr(Y, 'shape') else len(Y)

        K = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):
                x = X[i] if hasattr(X[i], '__len__') else [X[i]]
                y = Y[j] if hasattr(Y[j], '__len__') else [Y[j]]
                K[i, j] = self._compute_single_kernel(x, y)

        return K

    def _compute_single_kernel(self, x, y):
        """Вычисление ядра для одной пары точек"""
        raise NotImplementedError


class RBFKernel(KernelBase):
    """Гауссово (RBF) ядро"""

    def __init__(self, sigma=1.0, length_scale=1.0):
        super().__init__(sigma=sigma, length_scale=length_scale)
        self.sigma = sigma
        self.length_scale = length_scale

    def _compute_single_kernel(self, x, y):
        x = np.array(x)
        y = np.array(y)
        distance = np.linalg.norm(x - y)
        return self.sigma ** 2 * np.exp(-0.5 * (distance / self.length_scale) ** 2)


class MaternKernel(KernelBase):
    """Ядро Матерна для моделирования различной гладкости"""

    def __init__(self, nu=1.5, length_scale=1.0, sigma=1.0):
        super().__init__(nu=nu, length_scale=length_scale, sigma=sigma)
        self.nu = nu
        self.length_scale = length_scale
        self.sigma = sigma

    def _compute_single_kernel(self, x, y):
        x = np.array(x)
        y = np.array(y)
        d = np.linalg.norm(x - y) / self.length_scale

        if self.nu == 0.5:
            # Matern 1/2 (экспоненциальное)
            return self.sigma ** 2 * np.exp(-d)
        elif self.nu == 1.5:
            # Matern 3/2
            return self.sigma ** 2 * (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)
        elif self.nu == 2.5:
            # Matern 5/2
            return self.sigma ** 2 * (1 + np.sqrt(5) * d + (5 / 3) * d ** 2) * np.exp(-np.sqrt(5) * d)
        else:
            # Общий случай Матерна
            if d == 0:
                return self.sigma ** 2
            else:
                fraction = np.sqrt(2 * self.nu) * d
                return self.sigma ** 2 * (2 ** (1 - self.nu) / gamma(self.nu)) * \
                    (fraction ** self.nu) * kv(self.nu, fraction)


class SpectralMixtureKernel(KernelBase):
    """Спектральное смесевое ядро для периодических паттернов"""

    def __init__(self, num_mixtures=3, max_frequency=1.0):
        super().__init__(num_mixtures=num_mixtures, max_frequency=max_frequency)
        self.num_mixtures = num_mixtures
        self.max_frequency = max_frequency

        # Инициализация параметров смеси
        self.weights = nn.Parameter(torch.ones(num_mixtures) / num_mixtures)
        self.means = nn.Parameter(torch.linspace(0.1, max_frequency, num_mixtures))
        self.variances = nn.Parameter(torch.ones(num_mixtures) * 0.1)

    def _compute_single_kernel(self, x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        if len(x) != len(y):
            raise ValueError("Векторы должны иметь одинаковую размерность")

        kernel_value = 0.0
        delta = x - y

        for i in range(self.num_mixtures):
            weight = self.weights[i].item()
            mean = self.means[i].item()
            variance = self.variances[i].item()

            # Спектральное ядро для каждой смеси
            mixture_kernel = weight * np.exp(-2 * np.pi ** 2 * variance * np.dot(delta, delta)) * \
                             np.cos(2 * np.pi * mean * np.linalg.norm(delta))

            kernel_value += mixture_kernel

        return kernel_value


class FractionalMittagLefflerKernel(KernelBase):
    """Дробное ядро на основе функции Миттаг-Леффлера"""

    def __init__(self, q=0.7, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__(q=q, alpha=alpha, beta=beta, gamma=gamma)
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_single_kernel(self, x, y):
        x = np.array(x)
        y = np.array(y)
        distance = np.linalg.norm(x - y)

        # Используем функцию Миттаг-Леффлера как ядро
        argument = -self.gamma * (distance ** self.alpha)
        ml_value = mittag_leffler(argument, self.q)

        return self.beta * ml_value


class GraphDiffusionKernel(KernelBase):
    """Ядро графовой диффузии для структурных данных"""

    def __init__(self, diffusion_time=1.0, alpha=0.5):
        super().__init__(diffusion_time=diffusion_time, alpha=alpha)
        self.diffusion_time = diffusion_time
        self.alpha = alpha

    def _compute_laplacian_matrix(self, points):
        """Вычисление матрицы лапласиана графа"""
        n = len(points)
        W = np.zeros((n, n))  # Матрица смежности

        # Строим граф на основе попарных расстояний
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(points[i] - points[j])
                # Гауссово весовое ребро
                W[i, j] = np.exp(-distance ** 2 / 2.0)
                W[j, i] = W[i, j]

        # Степенная матрица
        D = np.diag(np.sum(W, axis=1))

        # Нормализованный лапласиан
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        return L

    def _compute_single_kernel(self, x, y):
        """Для графовых ядер нужен контекст всего набора данных"""
        # Этот метод требует переопределения для работы с графами
        # В реальной реализации мы бы кэшировали матрицу диффузии

        # Упрощенная версия: используем RBF для попарного сравнения
        distance = np.linalg.norm(np.array(x) - np.array(y))
        return np.exp(-distance ** 2 / (2 * self.diffusion_time))


class PeriodicKernel(KernelBase):
    """Периодическое ядро для сезонных паттернов"""

    def __init__(self, period=1.0, length_scale=1.0, sigma=1.0):
        super().__init__(period=period, length_scale=length_scale, sigma=sigma)
        self.period = period
        self.length_scale = length_scale
        self.sigma = sigma

    def _compute_single_kernel(self, x, y):
        x = np.array(x)
        y = np.array(y)

        # Периодическое расстояние
        if len(x) == 1 and len(y) == 1:
            # Для одномерных данных
            distance = np.abs(x[0] - y[0])
            periodic_distance = np.sin(np.pi * distance / self.period) ** 2
        else:
            # Для многомерных данных используем норму
            distance = np.linalg.norm(x - y)
            periodic_distance = np.sin(np.pi * distance / self.period) ** 2

        return self.sigma ** 2 * np.exp(-2 * periodic_distance / self.length_scale ** 2)


class RationalQuadraticKernel(KernelBase):
    """Рациональное квадратичное ядро (смесь RBF ядер)"""

    def __init__(self, alpha=1.0, length_scale=1.0, sigma=1.0):
        super().__init__(alpha=alpha, length_scale=length_scale, sigma=sigma)
        self.alpha = alpha
        self.length_scale = length_scale
        self.sigma = sigma

    def _compute_single_kernel(self, x, y):
        x = np.array(x)
        y = np.array(y)
        squared_distance = np.sum((x - y) ** 2)

        return self.sigma ** 2 * (1 + squared_distance / (2 * self.alpha * self.length_scale ** 2)) ** (-self.alpha)


class LinearKernel(KernelBase):
    """Линейное ядро (скалярное произведение)"""

    def __init__(self, constant=0.0, sigma=1.0):
        super().__init__(constant=constant, sigma=sigma)
        self.constant = constant
        self.sigma = sigma

    def _compute_single_kernel(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return self.sigma ** 2 * (np.dot(x, y) + self.constant)


class PolynomialKernel(KernelBase):
    """Полиномиальное ядро"""

    def __init__(self, degree=2, constant=1.0, sigma=1.0):
        super().__init__(degree=degree, constant=constant, sigma=sigma)
        self.degree = degree
        self.constant = constant
        self.sigma = sigma

    def _compute_single_kernel(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return self.sigma ** 2 * (np.dot(x, y) + self.constant) ** self.degree


class AdaptiveKernel(KernelBase):
    """Адаптивное ядро, которое обучает параметры на данных"""

    def __init__(self, initial_length_scale=1.0, learn_parameters=True):
        super().__init__(initial_length_scale=initial_length_scale, learn_parameters=learn_parameters)
        self.length_scale = nn.Parameter(torch.tensor(initial_length_scale))
        self.sigma = nn.Parameter(torch.tensor(1.0))
        self.learn_parameters = learn_parameters

    def _compute_single_kernel(self, x, y):
        x = np.array(x)
        y = np.array(y)
        distance = np.linalg.norm(x - y)

        length_scale = self.length_scale.item() if self.learn_parameters else self.initial_length_scale
        sigma = self.sigma.item() if self.learn_parameters else 1.0

        return sigma ** 2 * np.exp(-0.5 * (distance / length_scale) ** 2)

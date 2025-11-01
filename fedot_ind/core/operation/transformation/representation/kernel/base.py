import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma


class KernelBase(nn.Module):
    """Базовый класс для ядерных функций с расширенным функционалом"""

    def __init__(self, **kernel_params):
        super().__init__()
        self.kernel_params = kernel_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _is_torch_tensor(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device) if not isinstance(x, torch.Tensor) else x
        x = x.unsqueeze(-1) if x.ndim == 1 else x
        return x

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
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)
        distance = torch.norm(x - y)
        return (self.sigma ** 2 * torch.exp(-0.5 * (distance / self.length_scale) ** 2)).item()

    def _compute_batch_kernel(self, x, y):
        """
        Векторизованное вычисление ядра для батчей
        x: [..., dim], y: [..., dim]
        returns: [...] (такая же форма без последней размерности)
        """
        # Вычисляем попарные расстояния
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)
        distances = torch.norm(x - y, dim=-1)
        return self.sigma ** 2 * torch.exp(-0.5 * (distances / self.length_scale) ** 2)


class MaternKernel(KernelBase):
    """Ядро Матерна для моделирования различной гладкости"""

    def __init__(self, nu=1.5, length_scale=1.0, sigma=1.0):
        super().__init__(nu=nu, length_scale=length_scale, sigma=sigma)
        self.nu = nu
        self.length_scale = length_scale
        self.sigma = sigma

    def _compute_batch_kernel(self, x, y):
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)
        distances = torch.norm(x - y, dim=-1) / self.length_scale

        if self.nu == 0.5:
            # Matern 1/2
            return self.sigma ** 2 * torch.exp(-distances)
        elif self.nu == 1.5:
            # Matern 3/2
            sqrt3_d = torch.sqrt(torch.tensor(3.0)) * distances
            return self.sigma ** 2 * (1 + sqrt3_d) * torch.exp(-sqrt3_d)
        elif self.nu == 2.5:
            # Matern 5/2
            sqrt5_d = torch.sqrt(torch.tensor(5.0)) * distances
            return self.sigma ** 2 * (1 + sqrt5_d + (5 / 3) * distances ** 2) * torch.exp(-sqrt5_d)
        else:
            # Общий случай (аппроксимация)
            return self.sigma ** 2 * torch.exp(-distances ** self.nu)


class SpectralMixtureKernel(KernelBase):
    """Спектральное смесевое ядро с PyTorch бэкендом"""

    def __init__(self, num_mixtures=3, max_frequency=1.0):
        self.num_mixtures = num_mixtures
        self.max_frequency = max_frequency

        # Параметры как torch тензоры
        self.weights = nn.Parameter(torch.ones(num_mixtures) / num_mixtures)
        self.means = nn.Parameter(torch.linspace(0.1, max_frequency, num_mixtures))
        self.variances = nn.Parameter(torch.ones(num_mixtures) * 0.1)

    def _compute_batch_kernel(self, x, y):
        """
        Векторизованное вычисление спектрального ядра
        x: [..., dim], y: [..., dim]
        """
        delta = x - y  # [..., dim]

        # Вычисляем squared norm для каждой пары
        delta_sq = torch.sum(delta ** 2, dim=-1)  # [...]

        total_kernel = torch.zeros_like(delta_sq)

        for i in range(self.num_mixtures):
            weight = torch.sigmoid(self.weights[i])  # Ограничиваем (0,1)
            mean = torch.sigmoid(self.means[i]) * self.max_frequency
            variance = torch.nn.functional.softplus(self.variances[i])  # Положительная

            # Спектральное ядро для смеси
            mixture_kernel = weight * torch.exp(-2 * torch.pi ** 2 * variance * delta_sq) * \
                             torch.cos(2 * torch.pi * mean * torch.sqrt(delta_sq + 1e-8))

            total_kernel += mixture_kernel

        return total_kernel

    def _compute_single_kernel(self, x, y):
        x, y = self._is_torch_tensor(x, y)

        # Используем batch метод с добавлением размерностей
        x_expanded = x.unsqueeze(0) if x.ndim == 1 else x.unsqueeze(-2)
        y_expanded = y.unsqueeze(0) if y.ndim == 1 else y.unsqueeze(-2)

        kernel_val = self._compute_batch_kernel(x_expanded, y_expanded)
        return kernel_val.squeeze().item()


class FractionalMittagLefflerKernel(KernelBase):
    """Дробное ядро на основе функции Миттаг-Леффлера"""

    def __init__(self, q=0.7, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__(q=q, alpha=alpha, beta=beta, gamma=gamma)
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _mittag_leffler_torch(self, z, q, n_terms=50):
        """
        Функция Миттаг-Леффлера для PyTorch тензоров
        """
        result = torch.zeros_like(z)

        for k in range(n_terms):
            try:
                # Вычисляем k-й член ряда
                numerator = z ** k
                denominator = gamma(q * k + 1)

                # Используем torch.where для избежания деления на 0
                term = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(z))

                # Проверяем на NaN и Inf
                term = torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)

                result += term

                # Критерий остановки: если член ряда стал очень маленьким
                if torch.max(torch.abs(term)) < 1e-10:
                    break

            except:
                break

        return result

    def _compute_batch_kernel(self, x, y):
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)
        distances = torch.norm(x - y, dim=-1)
        argument = -self.gamma * (distances ** self.alpha)
        ml_values = self._mittag_leffler_torch(argument, self.q)
        return self.beta * ml_values


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

    def _compute_batch_kernel(self, x, y):
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)
        # Для простоты считаем одномерные данные
        if x.shape[-1] == 1 and y.shape[-1] == 1:
            distances = torch.abs(x - y)
        else:
            distances = torch.norm(x - y, dim=-1)

        periodic_distance = torch.sin(torch.pi * distances / self.period) ** 2
        return self.sigma ** 2 * torch.exp(-2 * periodic_distance / self.length_scale ** 2)


class RationalQuadraticKernel(KernelBase):
    """Рациональное квадратичное ядро (смесь RBF ядер)"""

    def __init__(self, alpha=1.0, length_scale=1.0, sigma=1.0):
        super().__init__(alpha=alpha, length_scale=length_scale, sigma=sigma)
        self.alpha = alpha
        self.length_scale = length_scale
        self.sigma = sigma

    def _compute_batch_kernel(self, x, y):
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)
        squared_distances = torch.sum((x - y) ** 2, dim=-1)
        return self.sigma ** 2 * (1 + squared_distances / (2 * self.alpha * self.length_scale ** 2)) ** (-self.alpha)


class LinearKernel(KernelBase):
    """Линейное ядро (скалярное произведение)"""

    def __init__(self, constant=0.0, sigma=1.0):
        super().__init__(constant=constant, sigma=sigma)
        self.constant = constant
        self.sigma = sigma

    def _compute_batch_kernel(self, x, y):
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)
        dot_products = torch.sum(x * y, dim=-1)
        return self.sigma ** 2 * (dot_products + self.constant)


class PolynomialKernel(KernelBase):
    """Полиномиальное ядро"""

    def __init__(self, degree=2, constant=1.0, sigma=1.0):
        super().__init__(degree=degree, constant=constant, sigma=sigma)
        self.degree = degree
        self.constant = constant
        self.sigma = sigma

    def _compute_batch_kernel(self, x, y):
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)
        dot_products = torch.sum(x * y, dim=-1)
        return self.sigma ** 2 * (dot_products + self.constant) ** self.degree


class AdaptiveKernel(KernelBase):
    """Адаптивное ядро с обучаемыми параметрами через PyTorch"""

    def __init__(self, initial_length_scale=1.0, learn_parameters=True):
        self.learn_parameters = learn_parameters

        if learn_parameters:
            self.log_length_scale = nn.Parameter(torch.log(torch.tensor(initial_length_scale)))
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(1.0)))
        else:
            self.length_scale = initial_length_scale
            self.sigma = 1.0

    def _compute_batch_kernel(self, x, y):
        if self.learn_parameters:
            length_scale = torch.exp(self.log_length_scale)
            sigma = torch.exp(self.log_sigma)
        else:
            length_scale = self.length_scale
            sigma = self.sigma

        distances = torch.norm(x - y, dim=-1)
        return sigma ** 2 * torch.exp(-0.5 * (distances / length_scale) ** 2)

    def _compute_single_kernel(self, x, y):
        x, y = self._is_torch_tensor(x), self._is_torch_tensor(y)

        x_expanded = x.unsqueeze(0) if x.ndim == 1 else x.unsqueeze(-2)
        y_expanded = y.unsqueeze(0) if y.ndim == 1 else y.unsqueeze(-2)

        kernel_val = self._compute_batch_kernel(x_expanded, y_expanded)
        return kernel_val.squeeze().item()

    def get_parameters(self):
        """Получить текущие значения параметров"""
        if self.learn_parameters:
            return {
                'length_scale': torch.exp(self.log_length_scale).item(),
                'sigma': torch.exp(self.log_sigma).item()
            }
        else:
            return {
                'length_scale': self.length_scale,
                'sigma': self.sigma
            }

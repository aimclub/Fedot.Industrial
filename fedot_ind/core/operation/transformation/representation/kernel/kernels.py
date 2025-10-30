from .base import *


class MultiKernelEnsemble(KernelBase):
    """Ансамбль различных ядерных функций"""

    def __init__(self, kernels=None, weights=None):
        super().__init__()

        if kernels is None:
            # Автоматическое создание разнообразного ансамбля
            self.kernels = [
                RBFKernel(sigma=1.0, length_scale=1.0),
                MaternKernel(nu=1.5, length_scale=1.0),
                PeriodicKernel(period=1.0, length_scale=1.0),
                RationalQuadraticKernel(alpha=1.0, length_scale=1.0),
                FractionalMittagLefflerKernel(q=0.7, alpha=1.0)
            ]
        else:
            self.kernels = kernels

        # Инициализация весов
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(self.kernels)) / len(self.kernels))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def _compute_single_kernel(self, x, y):
        """Вычисление комбинированного ядра"""
        total_kernel = 0.0

        for i, kernel in enumerate(self.kernels):
            weight = self.weights[i].item()
            kernel_value = kernel._compute_single_kernel(x, y)
            total_kernel += weight * kernel_value

        return total_kernel

    def compute_gram_matrix(self, trajectories):
        """Оптимизированное вычисление матрицы Грама"""
        n = len(trajectories)
        gram_matrix = np.zeros((n, n))

        # Предварительно вычисляем все парные ядра
        for i in range(n):
            for j in range(i, n):
                k_ij = self._compute_single_kernel(trajectories[i], trajectories[j])
                gram_matrix[i, j] = k_ij
                gram_matrix[j, i] = k_ij

        return gram_matrix

    def get_kernel_weights(self):
        """Получить текущие веса ядер"""
        return self.weights.detach().numpy()

    def set_kernel_weights(self, weights):
        """Установить веса ядер"""
        self.weights.data = torch.tensor(weights, dtype=torch.float32)


# Обновленный OccupationKernel с поддержкой новых ядер
class OccupationKernel(KernelBase):
    """Occupation Kernel с расширенным выбором базовых ядер"""

    def __init__(self, q=0.7, base_kernel_type='rbf', **kernel_params):
        """q -  дробный порядок интеграла ( по сути определяет как мы "взвешиваем историю" наблюдений)
        #####################################
        Режим "Короткой памяти"(0 < q < 0.5)
        Быстрое затухание весов, учитывает только последние 10-20% траектории
        Подходит для: высокочастотных данных, быстрых изменений
        #####################################
        Режим "Умеренной памяти" (0.5 ≤ q < 0.8)
        Плавное затухание весов
        Учитывает ~50% траектории
        Универсальный выбор для большинства задач
        #####################################
        Режим "Короткой памяти"(0.8 < q < 1.0)
        Медленное затухание весов
        Учитывает 70-90% траектории
        Подходит для: сезонных данных, долгосрочных зависимостей
        #####################################
        Быстрые финансовые данные (высокая волатильность)
        q_finance = 0.3
        Промышленные датчики (умеренная изменчивость)
        q_industrial = 0.5
        Погодные данные (сезонность)
        q_weather = 0.7
        Медицинские данные (долгосрочные тренды)
        q_medical = 0.8
        Геофизические данные (очень долгая память)
        q_geophysics = 0.9
        """
        super().__init__(q=q, **kernel_params)
        self.q = q
        # Выбор базового ядра
        if base_kernel_type == 'rbf':
            self.base_kernel = RBFKernel(**kernel_params)
        elif base_kernel_type == 'matern':
            self.base_kernel = MaternKernel(**kernel_params)
        elif base_kernel_type == 'periodic':
            self.base_kernel = PeriodicKernel(**kernel_params)
        elif base_kernel_type == 'fractional':
            self.base_kernel = FractionalMittagLefflerKernel(**kernel_params)
        elif base_kernel_type == 'rational_quadratic':
            self.base_kernel = RationalQuadraticKernel(**kernel_params)
        elif base_kernel_type == 'linear':
            self.base_kernel = LinearKernel(**kernel_params)
        elif base_kernel_type == 'polynomial':
            self.base_kernel = PolynomialKernel(**kernel_params)
        else:
            raise ValueError(f"Неизвестный тип ядра: {base_kernel_type}")

    def _compute_trajectory_kernel(self, traj1, traj2):
        """Вычисление ядра между двумя траекториями"""
        n1, n2 = len(traj1), len(traj2)
        total = 0.0

        for i in range(n1):
            for j in range(n2):
                # Вес с учетом дробного порядка
                weight_i = (n1 - i) ** (self.q - 1) / gamma(self.q)
                weight_j = (n2 - j) ** (self.q - 1) / gamma(self.q)

                # Базовое ядро между точками
                base_kernel_val = self.base_kernel._compute_single_kernel(traj1[i], traj2[j])

                total += weight_i * weight_j * base_kernel_val

        return total

    def compute_gram_matrix(self, trajectories):
        """Матрица Грама для набора траекторий"""
        n = len(trajectories)
        gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                k_ij = self._compute_trajectory_kernel(trajectories[i], trajectories[j])
                gram_matrix[i, j] = k_ij
                gram_matrix[j, i] = k_ij

        return gram_matrix

    def _compute_single_kernel(self, x, y):
        """Для совместимости с KernelBase"""
        # Если переданы траектории, используем occupation kernel
        if hasattr(x, '__len__') and hasattr(x[0], '__len__'):
            return self._compute_trajectory_kernel(x, y)
        else:
            # Иначе используем базовое ядро
            return self.base_kernel._compute_single_kernel(x, y)

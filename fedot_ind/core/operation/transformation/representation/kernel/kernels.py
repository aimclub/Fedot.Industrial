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
        self.kernel_dict = {'rbf': RBFKernel,
                            'linear': LinearKernel,
                            'polynomial': PolynomialKernel,
                            'matern': MaternKernel,
                            'periodic': PeriodicKernel,
                            'fractional': FractionalMittagLefflerKernel,
                            'rational_quadratic': RationalQuadraticKernel,
                            }
        # Выбор базового ядра
        self.base_kernel = self.kernel_dict[base_kernel_type](**kernel_params)
        self.q_selector = DataDrivenQSelector()

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
        self.q = self.q if self.q is not None else self.q_selector.analyze_and_suggest_q(trajectories)
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


class DataDrivenQSelector:
    """Выбор q на основе характеристик данных"""

    def suggest_q_based_on_autocorrelation(self, time_series):
        """Предложение q на основе автокорреляции"""
        from statsmodels.tsa.stattools import acf

        # Вычисляем автокорреляцию
        autocorr = acf(time_series, nlags=min(len(time_series) - 1, 50), fft=True)

        # Анализируем затухание автокорреляции
        lags = np.arange(1, len(autocorr))
        autocorr_vals = autocorr[1:]

        # Логарифмическая регрессия для определения типа затухания
        valid_mask = (autocorr_vals > 0) & (lags > 0)
        if np.sum(valid_mask) > 2:
            log_lags = np.log(lags[valid_mask])
            log_autocorr = np.log(autocorr_vals[valid_mask])

            slope, _ = np.polyfit(log_lags, log_autocorr, 1)

            # Маппинг наклона на q
            if slope > -0.3:  # Очень медленное затухание
                return 0.9
            elif slope > -0.7:  # Медленное затухание
                return 0.7
            elif slope > -1.2:  # Умеренное затухание
                return 0.5
            else:  # Быстрое затухание
                return 0.3
        else:
            return 0.7  # Значение по умолчанию

    def suggest_q_based_on_frequency(self, trajectories):
        """Предложение q на основе частотных характеристик"""
        from scipy import fftpack

        all_dominant_freqs = []

        for trajectory in trajectories:
            # FFT анализ
            fft_values = np.abs(fftpack.fft(trajectory))
            freqs = fftpack.fftfreq(len(trajectory))

            # Находим доминирующую частоту (исключая нулевую)
            idx = np.argsort(fft_values)[-2]  # Вторая по величине (первая - DC компонента)
            dominant_freq = np.abs(freqs[idx])
            all_dominant_freqs.append(dominant_freq)

        avg_frequency = np.mean(all_dominant_freqs)

        # Маппинг частоты на q
        if avg_frequency < 0.1:  # Низкочастотные данные
            return 0.9  # Долгая память
        elif avg_frequency < 0.3:  # Среднечастотные
            return 0.7  # Умеренная память
        else:  # Высокочастотные
            return 0.3  # Короткая память

    def analyze_and_suggest_q(self, trajectories, labels=None):
        """Комплексный анализ и предложение q"""

        # Анализ на одном примере
        example_series = trajectories[0] if hasattr(trajectories[0], '__len__') else trajectories

        q_autocorr = self.suggest_q_based_on_autocorrelation(example_series)
        q_frequency = self.suggest_q_based_on_frequency(trajectories[:10])

        # Усредняем рекомендации
        suggested_q = np.mean([q_autocorr, q_frequency])

        print(f"Рекомендации по q:")
        print(f"На основе автокорреляции: {q_autocorr:.2f}")
        print(f"На основе частотного анализа: {q_frequency:.2f}")
        print(f"Итоговая рекомендация: {suggested_q:.2f}")

        return suggested_q

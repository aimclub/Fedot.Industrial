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

    def __init__(self, q=0.7, kernel_type='rbf', **kernel_params):
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
        kernel_map = {
            'rbf': RBFKernel,
            'matern': MaternKernel,
            'periodic': PeriodicKernel,
            'fractional': FractionalMittagLefflerKernel,
            'rational_quadratic': RationalQuadraticKernel,
            'linear': LinearKernel,
            'polynomial': PolynomialKernel,
            'spectral_mixture': SpectralMixtureKernel,
            'graph_diffusion': GraphDiffusionKernel,
            'adaptive': AdaptiveKernel
        }
        # Выбор базового ядра
        self.base_kernel = kernel_map[kernel_type](**kernel_params)
        self.q_selector = DataDrivenQSelector()
        self._weight_cache = {}

    def _get_cached_weights(self, n):
        """Кэширование весов для длины траектории"""
        if n not in self._weight_cache:
            weights = torch.tensor([(n - i) ** (self.q - 1) / gamma(self.q)
                                    for i in range(n)], dtype=torch.float32)
            self._weight_cache[n] = weights
        return self._weight_cache[n]

    def _compute_trajectory_kernel(self, traj1, traj2):
        """Вычисление ядра между двумя траекториями"""
        traj1, traj2 = self._is_torch_tensor(traj1), self._is_torch_tensor(traj2)
        n1, n2 = len(traj1), len(traj2)
        weights1, weights2 = self._get_cached_weights(n1), self._get_cached_weights(n2)
        return self._compute_vectorized_kernel_single(traj1, traj2, weights1, weights2, n1, n2)

    def _compute_vectorized_kernel_single(self, traj1, traj2, weights1, weights2, len1, len2):
        """Векторизованное вычисление для одной пары траекторий"""
        # Расширяем для векторных операций
        traj1_expanded = traj1[:len1].unsqueeze(1)  # [len1, 1, dim]
        traj2_expanded = traj2[:len2].unsqueeze(0)  # [1, len2, dim]

        weights1_expanded = weights1[:len1].unsqueeze(1).to(self.device)  # [len1, 1]
        weights2_expanded = weights2[:len2].unsqueeze(0).to(self.device)  # [1, len2]

        # Вычисляем все попарные базовые ядра
        base_kernels = self.base_kernel._compute_batch_kernel(traj1_expanded, traj2_expanded)

        # Взвешенная сумма
        weighted_kernels = weights1_expanded * weights2_expanded * base_kernels
        return torch.sum(weighted_kernels).item()

    def compute_gram_matrix(self, trajectories, optimized=True):
        """
        Унифицированное вычисление матрицы Грама

        Args:
            trajectories: список траекторий
            optimized: использовать ли оптимизированную версию
        """
        if optimized and len(trajectories) > 50:  # Порог для оптимизированной версии
            return self._compute_gram_matrix_optimized(trajectories)
        else:
            return self._compute_gram_matrix_standard(trajectories)

    def _compute_gram_matrix_standard(self, trajectories):
        """Стандартное вычисление матрицы Грама"""
        n = len(trajectories)
        gram_matrix = torch.zeros((n, n), dtype=torch.float32)

        # Конвертируем все траектории
        torch_trajectories = [self._ensure_tensor(traj) for traj in trajectories]
        trajectory_lengths = [len(traj) for traj in torch_trajectories]

        # Предварительно вычисляем веса
        weight_matrices = self._precompute_weight_matrices(trajectory_lengths)

        for i in range(n):
            traj_i = torch_trajectories[i]
            weights_i = weight_matrices[i]
            len_i = trajectory_lengths[i]

            for j in range(i, n):  # Используем симметричность
                traj_j = torch_trajectories[j]
                weights_j = weight_matrices[j]
                len_j = trajectory_lengths[j]

                k_ij = self._compute_vectorized_kernel_single(
                    traj_i, traj_j, weights_i, weights_j, len_i, len_j
                )
                gram_matrix[i, j] = k_ij
                gram_matrix[j, i] = k_ij

        return gram_matrix.numpy()

    def _compute_gram_matrix_optimized(self, trajectories):
        """Оптимизированная версия для больших наборов данных"""
        n = len(trajectories)
        trajectory_lengths = [len(traj) for traj in trajectories]
        max_len = max(trajectory_lengths) if trajectory_lengths else 0
        weights_padded = torch.zeros((n, max_len), dtype=torch.float32)

        # Определяем размерность данных
        sample_traj = self._is_torch_tensor(trajectories[0])
        data_dim = sample_traj.shape[-1] if sample_traj.ndim > 1 else 1

        # Создаем padded тензор
        trajectories_padded = torch.zeros((n, max_len, data_dim), dtype=torch.float32)
        mask = torch.zeros((n, max_len), dtype=torch.bool)

        for i, traj in enumerate(trajectories):
            traj = self._is_torch_tensor(traj)
            actual_len = min(len(traj), max_len)
            trajectories_padded[i, :actual_len] = traj[:actual_len]
            mask[i, :actual_len] = True
            weights = self._get_cached_weights(len(traj))
            weights_padded[i, :actual_len] = weights[:actual_len]

        # # Предварительно вычисляем веса
        # for i, length in enumerate(trajectory_lengths):
        #     weights = self._get_cached_weights(length)
        #     actual_len = min(length, max_len)
        #     weights_padded[i, :actual_len] = weights[:actual_len]

        # Векторизованное вычисление
        gram_matrix = torch.zeros((n, n), dtype=torch.float32)

        for i in range(n):
            traj_i = trajectories_padded[i]  # [max_len, dim]
            weights_i = weights_padded[i]  # [max_len]
            mask_i = mask[i]  # [max_len]

            for j in range(i, n):
                traj_j = trajectories_padded[j]
                weights_j = weights_padded[j]
                mask_j = mask[j]

                # Вычисляем все попарные ядра
                traj_i_expanded = traj_i.unsqueeze(1)  # [max_len, 1, dim]
                traj_j_expanded = traj_j.unsqueeze(0)  # [1, max_len, dim]

                base_kernels = self.base_kernel._compute_batch_kernel(
                    traj_i_expanded, traj_j_expanded
                )  # [max_len, max_len]

                # Применяем веса и маски
                weights_ij = weights_i.unsqueeze(1) * weights_j.unsqueeze(0)
                mask_ij = mask_i.unsqueeze(1) & mask_j.unsqueeze(0)

                weighted_kernels = weights_ij * base_kernels
                weighted_kernels[~mask_ij] = 0

                k_ij = torch.sum(weighted_kernels)
                gram_matrix[i, j] = k_ij
                gram_matrix[j, i] = k_ij

        return gram_matrix.numpy()

    def _precompute_weight_matrices(self, lengths):
        """Предварительное вычисление матриц весов"""
        weight_matrices = []
        max_len = max(lengths) if lengths else 0

        for length in lengths:
            weights = torch.zeros(max_len, dtype=torch.float32)
            valid_weights = self._get_cached_weights(length)
            weights[:length] = valid_weights
            weight_matrices.append(weights)

        return weight_matrices

    def _compute_single_kernel(self, x, y):
        """Для совместимости с KernelBase"""
        return self._compute_trajectory_kernel(x, y)
    # def compute_gram_matrix(self, trajectories):
    #     """Матрица Грама для набора траекторий"""
    #     n = len(trajectories)
    #     gram_matrix = np.zeros((n, n))
    #     self.q = self.q if self.q is not None else self.q_selector.analyze_and_suggest_q(trajectories)
    #     for i in range(n):
    #         for j in range(i, n):
    #             k_ij = self._compute_trajectory_kernel(trajectories[i], trajectories[j])
    #             gram_matrix[i, j] = k_ij
    #             gram_matrix[j, i] = k_ij
    #
    #     return gram_matrix
    #
    # def _compute_single_kernel(self, x, y):
    #     """Для совместимости с KernelBase"""
    #     # Если переданы траектории, используем occupation kernel
    #     if hasattr(x, '__len__') and hasattr(x[0], '__len__'):
    #         return self._compute_trajectory_kernel(x, y)
    #     else:
    #         # Иначе используем базовое ядро
    #         return self.base_kernel._compute_single_kernel(x, y)
    #

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

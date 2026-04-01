
from sklearn.base import TransformerMixin
from scipy.special import gamma, roots_jacobi
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import eig

try:  # pragma: no cover - dependency is expected, but keep a safe fallback
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    from tqdm import tqdm  # type: ignore

import torch
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.special import gamma, roots_jacobi

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.policies import RegularizationPolicy
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.matrix_utils import MatrixComputationProgress

TQDM_FACTORY = tqdm
TQDM_WRITE = tqdm.write


class OKHSTransformer(TransformerMixin, BaseEstimator):
    """
    Трансформер для OKHS признаков с использованием физического времени
    и квадратур Гаусса-Якоби для учета сингулярностей (переписан на PyTorch).
    
    Отображает временные ряды из пространства траекторий R^{K \times d} 
    в конечномерное представление матрицы Грама R^{N \times N}.
    """

    def __init__(
            self,
            kernel,
            q=0.7,
            n_quad_points=20,
            dt=1.0,
            pairwise_block_size=64,
            regularization_policy=None,
            show_progress=False,
            progress_leave=False,
            verbose=False,
            device='cpu',
    ):
        """
        Инициализирует OKHSTransformer с параметрами ядра и квадратуры.
        
        Параметры
        ----------
        kernel : object
            Объект ядра с методами _compute_batch_kernel или _compute_single_kernel.
        q : float, default=0.7
            Параметр дробного порядка для взвешивания по времени.
            Контролирует поведение функции (T/2)^q в матрице Грама.
        n_quad_points : int, default=20
            Количество узлов квадратуры Гаусса-Якоби для интегрирования.
        dt : float, default=1.0
            Шаг дискретизации по времени для расчета длительности траектории.
        pairwise_block_size : int, default=64
            Размер блока для блочных вычислений матрицы Грама.
            Оптимизирует использование памяти при работе с большими выборками.
        regularization_policy : RegularizationPolicy, optional
            Политика регуляризации для матрицы Грама.
            Если не указана, используется политика по умолчанию.
        show_progress : bool, default=False
            Показывать ли индикатор прогресса при вычислении матриц.
        progress_leave : bool, default=False
            Оставлять ли индикатор прогресса на экране после завершения.
        verbose : bool, default=False
            Выводить ли детальные диагностические сообщения.
        """
        self.kernel = kernel
        self.q = q
        self.n_quad_points = n_quad_points
        self.dt = dt
        self.pairwise_block_size = pairwise_block_size
        self.C_q = 1.0 / gamma(q)
        self.regularization_policy = regularization_policy or RegularizationPolicy()
        self.show_progress = show_progress
        self.progress_leave = progress_leave
        self.verbose = verbose

        if device == 'cuda' and not torch.cuda.is_available():
            if self.verbose:
                print("Warning: CUDA is not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self._quad_cache = None
        
        print(f"OKHSTransformer initialized with device={self.device}, q={self.q}, n_quad_points={self.n_quad_points}, pairwise_block_size={self.pairwise_block_size}")

    def _get_trajectory_duration(self, trajectory):
        """
        Вычисляет длительность траектории на основе количества шагов.
        """
        n_steps = len(trajectory)
        return (n_steps - 1) * self.dt

    def _evaluate_trajectory_at_time(self, trajectory, t, T):
        """
        Вычисляет значение траектории в момент времени t методом линейной интерполяции.
        """
        if T <= 1e-14:
            return trajectory[0]
            
        n_steps = len(trajectory)
        t_idx = (t / T) * (n_steps - 1)
        
        idx = int(torch.floor(torch.tensor(t_idx)).item())
        idx = max(0, min(idx, n_steps - 2))
        
        alpha = t_idx - idx
        value = (1 - alpha) * trajectory[idx] + alpha * trajectory[idx + 1]
        return value

    def _get_jacobi_rule(self):
        """
        Получает узлы и веса Гаусса-Якоби для численного интегрирования.
        
        Результат кэшируется для повторного использования при нескольких вызовах.
        Используется параметр q из инициализации для специализированного взвешивания
        сингулярности на краях интервала.
        """
        if self._quad_cache is None:
            # SciPy используется исключительно для получения базисных констант,
            # дальше они переводятся в тензоры
            nodes, weights = roots_jacobi(self.n_quad_points, self.q - 1, 0)
            self._quad_cache = (
                torch.tensor(nodes, dtype=torch.float64),
                torch.tensor(weights, dtype=torch.float64)
            )
        return self._quad_cache

    def _normalize_trajectory(self,trajectory):
        """
        Нормализует траекторию в тензор PyTorch с типом float64 и правильной формой.
        
        Преобразует входные данные в torch.Tensor если необходимо,
        приводит к типу float64 и гарантирует 2D форму (n_steps, d).
        """
        if not isinstance(trajectory, torch.Tensor):
            normalized = torch.tensor(trajectory, dtype=torch.float64, device=self.device)
        else:
            normalized = trajectory.double().to(self.device)
            
        if normalized.ndim == 1:
            return normalized.view(-1, 1)
        return normalized

    def _evaluate_trajectory_at_nodes(self, trajectory, T):
        """
        Вычисляет значения траектории в узлах Гаусса-Якоби квадратурного правила.
        
        Для каждого узла квадратуры вычисляется интерполированное значение
        траектории в соответствующий момент времени.
        """
        normalized = self._normalize_trajectory(trajectory)
        n_steps = len(normalized)
        if T <= 1e-14 or n_steps <= 1:
            return normalized[0].unsqueeze(0).repeat(self.n_quad_points, 1)

        nodes, _ = self._get_jacobi_rule()
        nodes = nodes.to(normalized.device)
        
        t_nodes = T * (nodes + 1) / 2.0
        t_idx = (t_nodes / T) * (n_steps - 1)
        
        idx = torch.floor(t_idx).long()
        idx = torch.clamp(idx, 0, n_steps - 2)
        alpha = (t_idx - idx).unsqueeze(1)
        
        return (1.0 - alpha) * normalized[idx] + alpha * normalized[idx + 1]

    def _build_quadrature_cache(self, trajectories):
        """
        Создает кэш с предвычисленными значениями траекторий в узлах квадратуры.
        
        Кэш содержит значения всех траекторий в узлах квадратуры Гаусса-Якоби
        и масштабирующие коэффициенты (T/2)^q для каждой траектории.
        Это позволяет избежать повторных вычислений при расчете матрицы Грама.
        
        Параметры
        ----------
        trajectories : list of array-like
            Список траекторий для кэширования.
        
        Возвращает
        ----------
        dict
            Словарь с ключами:
            - 'values': тензор размера (n_trajectories, n_quad_points, d)
            - 'scales': тензор размера (n_trajectories,) со значениями (T/2)^q
            - 'weights': веса квадратуры размера (n_quad_points,)
        """
        _, weights = self._get_jacobi_rule()
        normalized_trajectories = [self._normalize_trajectory(traj) for traj in trajectories]

        device = normalized_trajectories[0].device if normalized_trajectories else torch.device('cpu')
        weights = weights.to(device)

        if not normalized_trajectories:
            return {
                "values": torch.zeros((0, self.n_quad_points, 0), dtype=torch.float64, device=device),
                "scales": torch.zeros(0, dtype=torch.float64, device=device),
                "weights": weights,
            }

        values = []
        scales = []
        for trajectory in normalized_trajectories:
            duration = self._get_trajectory_duration(trajectory)
            values.append(self._evaluate_trajectory_at_nodes(trajectory, duration))
            scales.append((duration / 2.0) ** self.q)

        return {
            "values": torch.stack(values, dim=0),
            "scales": torch.tensor(scales, dtype=torch.float64, device=device),
            "weights": weights,
        }

    def _compute_kernel_matrix_between_samples(self, left_samples, right_samples):
        """
        Вычисляет матрицу значений ядра между двумя наборами выборок.
        
        Использует специализированный batch-метод ядра если доступен,
        иначе вычисляет поточечно для каждой пары выборок.
        
        Параметры
        ----------
        left_samples : torch.Tensor
            Левый набор выборок размера (n_left, d).
        right_samples : torch.Tensor
            Правый набор выборок размера (n_right, d).
        
        Возвращает
        ----------
        torch.Tensor
            Матрица значений ядра размера (n_left, n_right) типа float64.
        """
        batch_kernel = getattr(self.kernel, "_compute_batch_kernel", None)
        if callable(batch_kernel):
            kernel_values = batch_kernel(
                left_samples.unsqueeze(1),
                right_samples.unsqueeze(0),
            )
            return kernel_values.double()

        kernel_matrix = torch.zeros((len(left_samples), len(right_samples)), dtype=torch.float64, device=left_samples.device)
        for left_index, left_sample in enumerate(left_samples):
            for right_index, right_sample in enumerate(right_samples):
                kernel_matrix[left_index, right_index] = self.kernel._compute_single_kernel(
                    left_sample,
                    right_sample,
                )
        return kernel_matrix

    def _compute_gram_entry_from_values(self, values_i, values_j, scale_i, scale_j, weights):
        """
        Вычисляет один элемент матрицы Грама OKHS из предвычисленных значений в узлах.
        
        Вычисляет следующую величину:
        G_ij = C_q^2 * scale_i * scale_j * Σ_p Σ_q w_p * K(ξ_p^i, ξ_q^j) * w_q
        
        где ξ - значения траекторий в узлах квадратуры, w - веса, K - ядро.
        """
        kernel_matrix = self._compute_kernel_matrix_between_samples(values_j, values_i)
        weighted_sum = torch.einsum('i,ij,j->', weights, kernel_matrix, weights).item()
        return (self.C_q ** 2) * weighted_sum * scale_i * scale_j

    def _resolve_pairwise_block_size(self, total_count):
        """
        Определяет финальный размер блока для блочных вычислений.
        
        Если pairwise_block_size равен 0 или None, возвращает общее количество.
        Иначе ограничивает размер между 1 и total_count.
        """
        if not self.pairwise_block_size:
            return max(1, total_count)
        return max(1, min(int(self.pairwise_block_size), total_count))

    def _count_symmetric_block_pairs(self, total_count, block_size):
        """
        Подсчитывает количество симметричных блоков для прогресса бара.
        
        Для симметричной матрицы подсчитывает блоки в верхнем треугольнике
        включая диагональ, с учетом блочной структуры.
        """
        count = 0
        for left_start in range(0, total_count, block_size):
            for _ in range(left_start, total_count, block_size):
                count += 1
        return count

    def _compute_gram_block_from_cache(self, left_values, right_values, left_scales, right_scales, weights):
        """
        Вычисляет блок матрицы Грама из предвычисленного кэша значений.
        
        Вычисляет прямоугольный блок матрицы Грама размера (n_left, n_right)
        используя значения траекторий в узлах квадратуры.
        
        Параметры
        ----------
        left_values : torch.Tensor
            Значения левого набора траекторий размера (n_left, n_quad_points, d).
        right_values : torch.Tensor
            Значения правого набора траекторий размера (n_right, n_quad_points, d).
        left_scales : torch.Tensor
            Масштабирующие коэффициенты левого набора размера (n_left,).
        right_scales : torch.Tensor
            Масштабирующие коэффициенты правого набора размера (n_right,).
        weights : torch.Tensor
            Веса квадратуры размера (n_quad_points,).
        
        Возвращает
        ----------
        torch.Tensor
            Блок матрицы Грама размера (n_left, n_right) типа float64.
        """
        batch_kernel = getattr(self.kernel, "_compute_batch_kernel", None)
        if callable(batch_kernel):
            kernel_values = batch_kernel(
                right_values.unsqueeze(2).unsqueeze(3),
                left_values.unsqueeze(0).unsqueeze(0),
            )
            # Приводим к типу float64 для сохранения обусловленности
            kernel_values = kernel_values.to(dtype=torch.float64, device=weights.device)
            weighted = torch.einsum('p,apbq,q->ab', weights, kernel_values, weights)
            return ((self.C_q ** 2) * weighted * right_scales.unsqueeze(1) * left_scales.unsqueeze(0)).T

        gram_block = torch.zeros((len(left_values), len(right_values)), dtype=torch.float64, device=weights.device)
        for left_index in range(len(left_values)):
            for right_index in range(len(right_values)):
                gram_block[left_index, right_index] = self._compute_gram_entry_from_values(
                    left_values[left_index],
                    right_values[right_index],
                    left_scales[left_index].item(),
                    right_scales[right_index].item(),
                    weights,
                )
        return gram_block

    def _compute_cross_gram_matrix_from_cache(self, left_cache, right_cache):
        """
        Вычисляет кросс-матрицу Грама между двумя наборами траекторий из кэша.
        
        Вычисляет матрицу K размера (n_left, n_right), где K[i,j] - скалярное
        произведение между i-й траекторией из left и j-й траекторией из right
        в OKHS норме.
        
        Параметры
        ----------
        left_cache : dict
            Кэш левого набора траекторий с ключами 'values', 'scales', 'weights'.
        right_cache : dict
            Кэш правого набора траекторий с ключами 'values', 'scales', 'weights'.
        
        Возвращает
        ----------
        torch.Tensor
            Кросс-матрица Грама размера (n_left, n_right) типа float64.
        """
        left_values = left_cache["values"]
        right_values = right_cache["values"]
        left_scales = left_cache["scales"]
        right_scales = right_cache["scales"]
        weights = left_cache["weights"]

        cross_gram_matrix = torch.zeros((len(left_values), len(right_values)), dtype=torch.float64, device=weights.device)
        block_size = self._resolve_pairwise_block_size(max(len(left_values), len(right_values)))
        for left_start in range(0, len(left_values), block_size):
            left_stop = min(left_start + block_size, len(left_values))
            left_slice = slice(left_start, left_stop)
            for right_start in range(0, len(right_values), block_size):
                right_stop = min(right_start + block_size, len(right_values))
                right_slice = slice(right_start, right_stop)
                cross_gram_matrix[left_slice, right_slice] = self._compute_gram_block_from_cache(
                    left_values[left_slice],
                    right_values[right_slice],
                    left_scales[left_slice],
                    right_scales[right_slice],
                    weights,
                )
        return cross_gram_matrix

    def _compute_gram_entry_jacobi(self, trajectory_i, trajectory_j):
        """
        Вычисляет один элемент матрицы Грама для двух траекторий напрямую.
        
        Этот метод используется для точечных вычислений без кэша.
        Автоматически вычисляет длительность обеих траекторий и применяет
        правило Гаусса-Якоби для интегрирования.
        """
        T_i = self._get_trajectory_duration(trajectory_i)
        T_j = self._get_trajectory_duration(trajectory_j)
        _, weights = self._get_jacobi_rule()

        vals_i = self._evaluate_trajectory_at_nodes(trajectory_i, T_i)
        vals_j = self._evaluate_trajectory_at_nodes(trajectory_j, T_j)
        scale_i = (T_i / 2.0) ** self.q
        scale_j = (T_j / 2.0) ** self.q
        
        device = vals_i.device
        return self._compute_gram_entry_from_values(
            vals_i,
            vals_j,
            scale_i,
            scale_j,
            weights.to(device),
        )
        
    def _compute_gram_matrix(self, trajectories):
        """
        Вычисляет полную матрицу Грама OKHS для набора траекторий.
        
        Матрица Грама размера (n, n) где n - количество траекторий.
        Вычисления выполняются блоками для эффективного использования памяти.
        Симметричность матрицы используется для сокращения вычислений.
        """
        quadrature_cache = self._build_quadrature_cache(trajectories)
        values = quadrature_cache["values"]
        scales = quadrature_cache["scales"]
        weights = quadrature_cache["weights"]
        n = len(values)
        
        gram_matrix = torch.zeros((n, n), dtype=torch.float64, device=weights.device)
        block_size = self._resolve_pairwise_block_size(n)
        progress = MatrixComputationProgress(
            enabled=True,
            matrix_name="gram",
            total_blocks=self._count_symmetric_block_pairs(n, block_size),
            n_items=n,
            leave=self.progress_leave,
        )
        try:
            for left_start in range(0, n, block_size):
                left_stop = min(left_start + block_size, n)
                left_slice = slice(left_start, left_stop)
                left_values = values[left_slice]
                left_scales = scales[left_slice]

                for right_start in range(left_start, n, block_size):
                    right_stop = min(right_start + block_size, n)
                    right_slice = slice(right_start, right_stop)
                    right_values = values[right_slice]
                    right_scales = scales[right_slice]
                    block = self._compute_gram_block_from_cache(
                        left_values,
                        right_values,
                        left_scales,
                        right_scales,
                        weights,
                    )
                    gram_matrix[left_slice, right_slice] = block
                    if left_start != right_start:
                        gram_matrix[right_slice, left_slice] = block.T
                    progress.update(left_start, left_stop, right_start, right_stop)
        finally:
            progress.close()

        self._last_quadrature_cache_ = quadrature_cache
        return gram_matrix

    def fit(self, train_trajectories, y=None):
        """
        Вычисляет и сохраняет матрицу Грама для набора тренировочных траекторий.
        
        После вычисления матрицы Грама применяется регуляризация и оценивается
        число обусловленности. Это необходимо для последующего преобразования
        тестовых данных методом transform().
        
        Параметры
        ----------
        train_trajectories : list of array-like
            Список тренировочных траекторий.
        y : array-like, optional
            Ложные метки, игнорируются (для совместимости со sklearn).
        
        Возвращает
        ----------
        self
            Возвращает себя для методов цепочки (chaining).
        """
        self.train_trajectories_ = train_trajectories
        raw_gram_matrix = self._compute_gram_matrix(train_trajectories)
        
        # Инфраструктурная проверка формы, предполагается, что validate_square_matrix_shape 
        # принимает тензоры и проверяет tensor.shape (работает так же, как с numpy)
        # validate_square_matrix_shape(raw_gram_matrix, "Gram matrix") 
        
        self._train_quadrature_cache_ = self._last_quadrature_cache_

        self.gram_condition_number_ = torch.linalg.cond(raw_gram_matrix).item()
        regularization = self.regularization_policy.base_jitter * torch.eye(len(train_trajectories), dtype=torch.float64, device=raw_gram_matrix.device)
        self.gram_matrix_ = raw_gram_matrix + regularization
        return self

    def transform(self, test_trajectories):
        """
        Преобразует набор тестовых траекторий в OKHS признаки.
        
        Вычисляет кросс-матрицу Грама между тестовыми и тренировочными
        траекториями, затем решает линейную систему для получения коэффициентов
        представления в базисе OKHS.
        
        Параметры
        ----------
        test_trajectories : list of array-like
            Список тестовых траекторий для преобразования.
        
        Возвращает
        ----------
        torch.Tensor
            OKHS признаки размера (n_test, n_train) типа float64,
            где строка i содержит коэффициенты разложения i-й тестовой
            траектории по базису тренировочных траекторий.
        
        Вызывает исключение
        ------
        RuntimeError
            Если матрица Грама сингулярна и fallback_solver != 'pinv'.
        """
        train_cache = getattr(self, "_train_quadrature_cache_", None)
        if train_cache is None:
            train_cache = self._build_quadrature_cache(self.train_trajectories_)
            self._train_quadrature_cache_ = train_cache

        test_cache = self._build_quadrature_cache(test_trajectories)
        K_test_train = self._compute_cross_gram_matrix_from_cache(test_cache, train_cache)

        should_use_pinv = self.gram_condition_number_ > self.regularization_policy.condition_threshold
        if should_use_pinv and self.regularization_policy.fallback_solver == "pinv":
            c = K_test_train @ torch.linalg.pinv(self.gram_matrix_)
        else:
            try:
                # torch.linalg.solve бросает RuntimeError вместо LinAlgError при сингулярности матрицы
                c = torch.linalg.solve(self.gram_matrix_, K_test_train.T).T
            except RuntimeError:
                if self.regularization_policy.fallback_solver != "pinv":
                    raise
                c = K_test_train @ torch.linalg.pinv(self.gram_matrix_)
            
        return c
    


    
from dataclasses import dataclass
import time
import warnings

import numpy as np
from sklearn.base import TransformerMixin
from scipy.special import gamma, roots_jacobi
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import eig

try:  # pragma: no cover - dependency is expected, but keep a safe fallback
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    from tqdm import tqdm  # type: ignore

try:
    from pymittagleffler import mittag_leffler
except ImportError:  # pragma: no cover - fallback for local environments without pymittagleffler
    from fedot_ind.core.operation.transformation.representation.kernel.utils import mittag_leffler

TQDM_FACTORY = tqdm
TQDM_WRITE = tqdm.write


@dataclass(frozen=True)
class RegularizationPolicy:
    base_jitter: float = 1e-8
    condition_threshold: float = 1e10
    fallback_solver: str = "pinv"


@dataclass(frozen=True)
class StabilityPolicy:
    threshold: float = 0.0
    drop_positive_real_modes: bool = True
    sorting_strategy: str = "abs_desc"


@dataclass
class MatrixComputationProgress:
    enabled: bool
    matrix_name: str
    total_blocks: int
    n_items: int
    leave: bool = False
    bar: object = None
    started_at: float = 0.0
    completed_blocks: int = 0

    def __post_init__(self):
        self.started_at = time.perf_counter()
        if self.enabled:
            self.bar = TQDM_FACTORY(
                total=self.total_blocks,
                desc=f"okhs:{self.matrix_name}",
                unit="block",
                dynamic_ncols=True,
                leave=self.leave,
            )
            TQDM_WRITE(
                f"[okhs] building {self.matrix_name} matrix for {self.n_items} trajectories in {self.total_blocks} blocks"
            )

    def update(self, left_start, left_stop, right_start, right_stop):
        self.completed_blocks += 1
        if not self.enabled or self.bar is None:
            return
        elapsed_seconds = max(time.perf_counter() - self.started_at, 1e-9)
        blocks_per_second = self.completed_blocks / elapsed_seconds
        remaining_blocks = max(self.total_blocks - self.completed_blocks, 0)
        eta_seconds = remaining_blocks / max(blocks_per_second, 1e-9)
        if hasattr(self.bar, "set_postfix"):
            self.bar.set_postfix(
                {
                    "left": f"{left_start}:{left_stop}",
                    "right": f"{right_start}:{right_stop}",
                    "elapsed_s": f"{elapsed_seconds:.1f}",
                    "blk_s": f"{blocks_per_second:.2f}",
                    "eta_s": f"{eta_seconds:.1f}",
                },
                refresh=False,
            )
        if hasattr(self.bar, "update"):
            self.bar.update(1)

    def close(self):
        elapsed_seconds = max(time.perf_counter() - self.started_at, 0.0)
        blocks_per_second = self.completed_blocks / max(elapsed_seconds, 1e-9) if self.completed_blocks else 0.0
        if self.enabled:
            TQDM_WRITE(
                f"[okhs] finished {self.matrix_name} matrix "
                f"elapsed_s={elapsed_seconds:.2f} blocks={self.completed_blocks} blk_s={blocks_per_second:.2f}"
            )
        if self.enabled and self.bar is not None and hasattr(self.bar, "close"):
            self.bar.close()


def validate_square_matrix_shape(matrix, label):
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"{label} must be square, got shape {matrix.shape}.")


def validate_liouville_shapes(gram_matrix, liouville_matrix):
    validate_square_matrix_shape(gram_matrix, "Gram matrix")
    validate_square_matrix_shape(liouville_matrix, "Liouville matrix")
    if gram_matrix.shape != liouville_matrix.shape:
        raise ValueError(
            "Gram and Liouville matrices must have identical shapes, "
            f"got {gram_matrix.shape} and {liouville_matrix.shape}."
        )


def validate_initial_coefficient_feasibility(initial_trajectory, n_modes):
    initial_trajectory = np.asarray(initial_trajectory)
    if initial_trajectory.ndim != 2:
        raise ValueError("initial_trajectory must have shape (K, n_features).")

    n_points, n_features = initial_trajectory.shape
    available_equations = n_points * n_features
    if available_equations < n_modes:
        raise ValueError(
            f"Insufficient data for coefficient fit: {available_equations} equations < {n_modes} modes."
        )


def select_stable_modes(eigenvalues, stability_policy, stability_threshold=None):
    if not stability_policy.drop_positive_real_modes:
        return np.ones(len(eigenvalues), dtype=bool)

    threshold = (
        stability_policy.threshold
        if stability_threshold is None
        else stability_threshold
    )
    return np.real(eigenvalues) < threshold


def sort_eigendecomposition(eigenvalues, eigenvectors, sorting_strategy):
    if sorting_strategy == "real_desc":
        order = np.argsort(np.real(eigenvalues))[::-1]
    else:
        order = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[order], eigenvectors[:, order]

class OKHSTransformer(TransformerMixin, BaseEstimator):
    """
    Трансформер для OKHS признаков с использованием физического времени
    и квадратур Гаусса-Якоби для учета сингулярностей.
    
    Матрица Грама вычисляется как:
    G_{i,j} = C_q^2 * ∫∫_{[0,T]²} (T - τ)^{q-1} (T - t)^{q-1} K(ξ_j(t), ξ_i(τ)) dt dτ
    
    С использованием квадратур Якоби сингулярный вес (T-t)^{q-1} учитывается
    в весах квадратуры w_k, что дает точное интегрирование для полиномиальных ядер.
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
    ):
        """
        Параметры:
        -----------
        kernel : KernelBase
            Ядро (например, RBFKernel). Должно иметь метод _compute_single_kernel(x, y).
        
        q : float (0 < q ≤ 1)
            Порядок дробной производной.
        
        n_quad_points : int
            Количество точек для метода 'jacobi'.
            
        dt : float
            Шаг дискретизации времени траекторий. Используется для вычисления T.
            Если данные не имеют временной метки, dt=1.0 означает, что T = n_steps.
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
        
        # Кэш для узлов и весов квадратуры
        self._quad_cache = None

    def _get_trajectory_duration(self, trajectory):
        """
        Возвращает физическую длительность траектории T.
        """
        n_steps = len(trajectory)
        # T - это интервал времени от 0 до конца.
        # Если 5 точек с шагом 0.1, то T = 0.4 (t: 0.0, 0.1, 0.2, 0.3, 0.4)
        return (n_steps - 1) * self.dt

    def _evaluate_trajectory_at_time(self, trajectory, t, T):
        """
        Интерполяция траектории в физический момент времени t ∈ [0, T].
        """
        if T <= 1e-14: # Защита от деления на ноль для вырожденных траекторий
            return trajectory[0]
            
        # Нормализуем время к индексу массива
        # t / T дает долю пути (0..1), умножаем на (N-1) чтобы получить индекс
        n_steps = len(trajectory)
        t_idx = (t / T) * (n_steps - 1)
        
        idx = int(np.floor(t_idx))
        idx = np.clip(idx, 0, n_steps - 2)
        
        alpha = t_idx - idx
        
        # Линейная интерполяция
        value = (1 - alpha) * trajectory[idx] + alpha * trajectory[idx + 1]
        return value

    def _get_jacobi_rule(self):
        """
        Возвращает узлы (x) и веса (w) квадратуры Гаусса-Якоби для интеграла:
        ∫_{-1}^{1} (1-x)^alpha (1+x)^beta f(x) dx.
        
        Нам нужно интегрировать (T - t)^(q-1).
        При замене переменной t = T(x+1)/2, член (T-t) пропорционален (1-x).
        Поэтому alpha = q - 1, beta = 0.
        """
        if self._quad_cache is None:
            # alpha=q-1, beta=0. 
            # Функция roots_jacobi возвращает узлы для веса (1-x)^alpha (1+x)^beta
            self._quad_cache = roots_jacobi(self.n_quad_points, self.q - 1, 0)
        return self._quad_cache

    @staticmethod
    def _normalize_trajectory(trajectory):
        normalized = np.asarray(trajectory, dtype=float)
        if normalized.ndim == 1:
            return normalized.reshape(-1, 1)
        return normalized

    def _evaluate_trajectory_at_nodes(self, trajectory, T):
        normalized = self._normalize_trajectory(trajectory)
        n_steps = len(normalized)
        if T <= 1e-14 or n_steps <= 1:
            return np.repeat(normalized[[0]], self.n_quad_points, axis=0)

        nodes, _ = self._get_jacobi_rule()
        t_nodes = T * (nodes + 1) / 2.0
        t_idx = (t_nodes / T) * (n_steps - 1)
        idx = np.floor(t_idx).astype(int)
        idx = np.clip(idx, 0, n_steps - 2)
        alpha = (t_idx - idx)[:, None]
        return (1.0 - alpha) * normalized[idx] + alpha * normalized[idx + 1]

    def _build_quadrature_cache(self, trajectories):
        _, weights = self._get_jacobi_rule()
        normalized_trajectories = [self._normalize_trajectory(trajectory) for trajectory in trajectories]

        if not normalized_trajectories:
            return {
                "values": np.zeros((0, self.n_quad_points, 0), dtype=float),
                "scales": np.zeros(0, dtype=float),
                "weights": np.asarray(weights, dtype=float),
            }

        values = []
        scales = []
        for trajectory in normalized_trajectories:
            duration = self._get_trajectory_duration(trajectory)
            values.append(self._evaluate_trajectory_at_nodes(trajectory, duration))
            scales.append((duration / 2.0) ** self.q)

        return {
            "values": np.stack(values, axis=0),
            "scales": np.asarray(scales, dtype=float),
            "weights": np.asarray(weights, dtype=float),
        }

    def _compute_kernel_matrix_between_samples(self, left_samples, right_samples):
        batch_kernel = getattr(self.kernel, "_compute_batch_kernel", None)
        if callable(batch_kernel):
            kernel_values = batch_kernel(
                np.asarray(left_samples, dtype=float)[:, None, :],
                np.asarray(right_samples, dtype=float)[None, :, :],
            )
            if hasattr(kernel_values, "detach"):
                kernel_values = kernel_values.detach().cpu().numpy()
            return np.asarray(kernel_values, dtype=float)

        kernel_matrix = np.zeros((len(left_samples), len(right_samples)), dtype=float)
        for left_index, left_sample in enumerate(left_samples):
            for right_index, right_sample in enumerate(right_samples):
                kernel_matrix[left_index, right_index] = self.kernel._compute_single_kernel(
                    left_sample,
                    right_sample,
                )
        return kernel_matrix

    def _compute_gram_entry_from_values(self, values_i, values_j, scale_i, scale_j, weights):
        kernel_matrix = self._compute_kernel_matrix_between_samples(values_j, values_i)
        weighted_sum = float(weights @ kernel_matrix @ weights)
        return (self.C_q ** 2) * weighted_sum * scale_i * scale_j

    def _resolve_pairwise_block_size(self, total_count):
        if not self.pairwise_block_size:
            return max(1, total_count)
        return max(1, min(int(self.pairwise_block_size), total_count))

    def _count_symmetric_block_pairs(self, total_count, block_size):
        count = 0
        for left_start in range(0, total_count, block_size):
            for _ in range(left_start, total_count, block_size):
                count += 1
        return count

    def _compute_gram_block_from_cache(self, left_values, right_values, left_scales, right_scales, weights):
        batch_kernel = getattr(self.kernel, "_compute_batch_kernel", None)
        if callable(batch_kernel):
            kernel_values = batch_kernel(
                np.asarray(right_values, dtype=float)[:, :, None, None, :],
                np.asarray(left_values, dtype=float)[None, None, :, :, :],
            )
            if hasattr(kernel_values, "detach"):
                kernel_values = kernel_values.detach().cpu().numpy()
            kernel_values = np.asarray(kernel_values, dtype=float)
            weighted = np.einsum('p,apbq,q->ab', weights, kernel_values, weights)
            return ((self.C_q ** 2) * weighted * right_scales[:, None] * left_scales[None, :]).T

        gram_block = np.zeros((len(left_values), len(right_values)), dtype=float)
        for left_index in range(len(left_values)):
            for right_index in range(len(right_values)):
                gram_block[left_index, right_index] = self._compute_gram_entry_from_values(
                    left_values[left_index],
                    right_values[right_index],
                    left_scales[left_index],
                    right_scales[right_index],
                    weights,
                )
        return gram_block

    def _compute_cross_gram_matrix_from_cache(self, left_cache, right_cache):
        left_values = left_cache["values"]
        right_values = right_cache["values"]
        left_scales = left_cache["scales"]
        right_scales = right_cache["scales"]
        weights = left_cache["weights"]

        cross_gram_matrix = np.zeros((len(left_values), len(right_values)), dtype=float)
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
        ???????????????????? ???????????????? ?????????? ???????????????????? ????????????-??????????.
        
        ???????????????? ????????: I = ???_0^T (T-t)^(q-1) f(t) dt.
        ????????????: t = T(x+1)/2, dt = (T/2)dx.
        (T - t) = T - T(x+1)/2 = T(1 - (x+1)/2) = T(1-x)/2.
        
        ?????????? (T-t)^(q-1) = (T/2)^(q-1) * (1-x)^(q-1).
        
        I = ???_{-1}^1 (T/2)^(q-1) (1-x)^(q-1) f(T(x+1)/2) * (T/2) dx
          = (T/2)^q ???_{-1}^1 (1-x)^(q-1) f(...) dx
          
        ???????????????? ???????????????????? ??????????: Sum w_k * f(x_k). 
        ?????? (1-x)^(q-1) ?????? "??????????" ?? w_k.
        ???????????????? ???????????? ?????????????????? (T/2)^q.
        """
        T_i = self._get_trajectory_duration(trajectory_i)
        T_j = self._get_trajectory_duration(trajectory_j)
        _, weights = self._get_jacobi_rule()

        vals_i = self._evaluate_trajectory_at_nodes(trajectory_i, T_i)
        vals_j = self._evaluate_trajectory_at_nodes(trajectory_j, T_j)
        scale_i = (T_i / 2.0) ** self.q
        scale_j = (T_j / 2.0) ** self.q
        return self._compute_gram_entry_from_values(
            vals_i,
            vals_j,
            scale_i,
            scale_j,
            np.asarray(weights, dtype=float),
        )
        

    def _compute_gram_matrix(self, trajectories):
        quadrature_cache = self._build_quadrature_cache(trajectories)
        values = quadrature_cache["values"]
        scales = quadrature_cache["scales"]
        weights = quadrature_cache["weights"]
        n = len(values)
        gram_matrix = np.zeros((n, n), dtype=float)
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
        Обучение трансформера: вычисление матрицы Грама.
        
        Parameters
        ----------
        train_trajectories : list of array-like
            Список обучающих траекторий
        """
        self.train_trajectories_ = train_trajectories
        raw_gram_matrix = self._compute_gram_matrix(train_trajectories)
        validate_square_matrix_shape(raw_gram_matrix, "Gram matrix")
        self._train_quadrature_cache_ = self._last_quadrature_cache_

        self.gram_condition_number_ = np.linalg.cond(raw_gram_matrix)
        regularization = self.regularization_policy.base_jitter * np.eye(len(train_trajectories))
        self.gram_matrix_ = raw_gram_matrix + regularization
        return self

    def transform(self, test_trajectories):
        """
        Вычисляет координаты новых (тестовых) траекторий в базисе обучающих.
        
        Parameters
        ----------
        test_trajectories : list of array-like
            Список тестовых траекторий
        """
        n_train = len(self.train_trajectories_)
        n_test = len(test_trajectories)

        train_cache = getattr(self, "_train_quadrature_cache_", None)
        if train_cache is None:
            train_cache = self._build_quadrature_cache(self.train_trajectories_)
            self._train_quadrature_cache_ = train_cache

        test_cache = self._build_quadrature_cache(test_trajectories)
        K_test_train = self._compute_cross_gram_matrix_from_cache(test_cache, train_cache)

        # c = (G^-1 K^T)^T = K G^-1
        # G c^T = K^T -> решаем для каждой строки
        should_use_pinv = self.gram_condition_number_ > self.regularization_policy.condition_threshold
        if should_use_pinv and self.regularization_policy.fallback_solver == "pinv":
            c = K_test_train @ np.linalg.pinv(self.gram_matrix_)
        else:
            try:
                c = np.linalg.solve(self.gram_matrix_, K_test_train.T).T
            except np.linalg.LinAlgError:
                if self.regularization_policy.fallback_solver != "pinv":
                    raise
                c = K_test_train @ np.linalg.pinv(self.gram_matrix_)
            
        return c
    

class FractionalLiouvilleOperator(BaseEstimator):
    """
    Оператор Лиувилля дробного порядка с использованием квадратур Якоби.
    
    Реализует конечномерное представление оператора P A_{f,q} P.
    Элементы матрицы вычисляются через однократный интеграл с сингулярным весом:
    
    A_{ij} = <A* mu_i, mu_j> 
           = C_q * ∫_0^T (T-τ)^{q-1} [K(ξ_j(τ), ξ_i(T)) - K(ξ_j(τ), ξ_i(0))] dτ
           
    Интеграл вычисляется точно с помощью квадратур Гаусса-Якоби для веса (1-x)^{q-1}.
    """

    def __init__(
            self,
            okhs_transformer,
            n_quad_points=20,
            pairwise_block_size=64,
            regularization_policy=None,
            stability_policy=None,
            verbose=False,
    ):
        """
        Parameters
        ----------
        okhs_transformer : OKHSTransformer
            Обученный экземпляр OKHSTransformer. Должен иметь атрибуты q, C_q, dt.
            
        n_quad_points : int
            Число точек квадратуры Якоби для вычисления элементов оператора.
        """
        self.okhs = okhs_transformer
        self.n_quad_points = n_quad_points
        self.pairwise_block_size = pairwise_block_size
        self.regularization_policy = regularization_policy or RegularizationPolicy()
        self.stability_policy = stability_policy or StabilityPolicy()
        self.show_progress = getattr(okhs_transformer, "show_progress", False)
        self.progress_leave = getattr(okhs_transformer, "progress_leave", False)
        self.verbose = verbose
        
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.liouville_matrix_ = None
        
        # Кэш для квадратур
        self._quad_cache = None

    def _get_jacobi_rule(self):
        """???????????????????? ???????? ?? ???????? ?????? ???????? (1-x)^{q-1}."""
        if self._quad_cache is None:
            # alpha = q - 1, beta = 0
            # weight function: (1-x)^alpha * (1+x)^0
            q = self.okhs.q
            self._quad_cache = roots_jacobi(self.n_quad_points, q - 1, 0)
        return self._quad_cache

    def _build_liouville_cache(self, trajectories):
        quadrature_cache = getattr(self.okhs, '_train_quadrature_cache_', None)
        if quadrature_cache is None or len(quadrature_cache['values']) != len(trajectories):
            quadrature_cache = self.okhs._build_quadrature_cache(trajectories)

        start_points = []
        end_points = []
        for trajectory in trajectories:
            normalized = self.okhs._normalize_trajectory(trajectory)
            start_points.append(normalized[0])
            end_points.append(normalized[-1])

        return {
            'quadrature': quadrature_cache,
            'start_points': np.asarray(start_points, dtype=float),
            'end_points': np.asarray(end_points, dtype=float),
        }

    def _compute_liouville_entry(self, traj_i, traj_j):
        """
        ?????????????????? ?????????????? ?????????????? A_{ij}.
        """
        T_i = self.okhs._get_trajectory_duration(traj_i)
        T_j = self.okhs._get_trajectory_duration(traj_j)

        if T_i <= 1e-14 or T_j <= 1e-14:
            return 0.0

        _, weights = self._get_jacobi_rule()
        values_j = self.okhs._evaluate_trajectory_at_nodes(traj_j, T_j)
        normalized_i = self.okhs._normalize_trajectory(traj_i)
        end_point_i = normalized_i[-1]
        start_point_i = normalized_i[0]
        scale_j = (T_j / 2.0) ** self.okhs.q

        kernel_to_end = self.okhs._compute_kernel_matrix_between_samples(values_j, end_point_i[None, :]).reshape(-1)
        kernel_to_start = self.okhs._compute_kernel_matrix_between_samples(values_j, start_point_i[None, :]).reshape(-1)
        return self.okhs.C_q * scale_j * float(np.asarray(weights, dtype=float) @ (kernel_to_end - kernel_to_start))

    def _compute_liouville_block_from_cache(self, left_start_points, left_end_points, right_values, right_scales,
                                            weights):
        batch_kernel = getattr(self.okhs.kernel, "_compute_batch_kernel", None)
        if callable(batch_kernel):
            kernel_to_end = batch_kernel(
                np.asarray(right_values, dtype=float)[:, :, None, :],
                np.asarray(left_end_points, dtype=float)[None, None, :, :],
            )
            kernel_to_start = batch_kernel(
                np.asarray(right_values, dtype=float)[:, :, None, :],
                np.asarray(left_start_points, dtype=float)[None, None, :, :],
            )
            if hasattr(kernel_to_end, "detach"):
                kernel_to_end = kernel_to_end.detach().cpu().numpy()
            if hasattr(kernel_to_start, "detach"):
                kernel_to_start = kernel_to_start.detach().cpu().numpy()
            weighted_end = np.einsum('p,apb->ab', weights, np.asarray(kernel_to_end, dtype=float))
            weighted_start = np.einsum('p,apb->ab', weights, np.asarray(kernel_to_start, dtype=float))
            return (self.okhs.C_q * right_scales[:, None] * (weighted_end - weighted_start)).T

        liouville_block = np.zeros((len(left_start_points), len(right_values)), dtype=float)
        for left_index in range(len(left_start_points)):
            for right_index in range(len(right_values)):
                kernel_to_end = self.okhs._compute_kernel_matrix_between_samples(
                    right_values[right_index],
                    left_end_points[left_index][None, :],
                ).reshape(-1)
                kernel_to_start = self.okhs._compute_kernel_matrix_between_samples(
                    right_values[right_index],
                    left_start_points[left_index][None, :],
                ).reshape(-1)
                liouville_block[left_index, right_index] = self.okhs.C_q * right_scales[right_index] * float(
                    weights @ (kernel_to_end - kernel_to_start)
                )
        return liouville_block

    def fit(self, trajectories=None):
        """
        Строит матрицу оператора и решает обобщенную задачу на собственные значения.
        """
        if trajectories is None:
            if not hasattr(self.okhs, 'train_trajectories_'):
                raise ValueError("OKHSTransformer must be fitted first.")
            trajectories = self.okhs.train_trajectories_
        
        n_traj = len(trajectories)
        liouville_cache = self._build_liouville_cache(trajectories)
        values = liouville_cache['quadrature']['values']
        scales = liouville_cache['quadrature']['scales']
        weights = liouville_cache['quadrature']['weights']
        start_points = liouville_cache['start_points']
        end_points = liouville_cache['end_points']
        self.liouville_matrix_ = np.zeros((n_traj, n_traj), dtype=float)

        if self.verbose:
            print(f"Computing Liouville matrix ({n_traj}x{n_traj}) using Jacobi quadratures...")
        block_size = n_traj if not self.pairwise_block_size else max(1, min(int(self.pairwise_block_size), n_traj))
        progress = MatrixComputationProgress(
            enabled=self.show_progress,
            matrix_name="liouville",
            total_blocks=int(np.ceil(n_traj / block_size)) ** 2,
            n_items=n_traj,
            leave=self.progress_leave,
        )
        try:
            for left_start in range(0, n_traj, block_size):
                left_stop = min(left_start + block_size, n_traj)
                left_slice = slice(left_start, left_stop)

                for right_start in range(0, n_traj, block_size):
                    right_stop = min(right_start + block_size, n_traj)
                    right_slice = slice(right_start, right_stop)
                    self.liouville_matrix_[left_slice, right_slice] = self._compute_liouville_block_from_cache(
                        start_points[left_slice],
                        end_points[left_slice],
                        values[right_slice],
                        scales[right_slice],
                        weights,
                    )
                    progress.update(left_start, left_stop, right_start, right_stop)
        finally:
            progress.close()
                
        G = self.okhs.gram_matrix_
        validate_liouville_shapes(G, self.liouville_matrix_)

        if self.verbose:
            print("Solving generalized eigenvalue problem A v = lambda G v...")
        try:
            # Решаем A * v = lambda * G * v
            eigenvalues, eigenvectors = eig(self.liouville_matrix_, G)
        except Exception as e:
            if self.regularization_policy.fallback_solver != "pinv":
                raise
            if self.verbose:
                print(f"Generalized eig failed ({e}), using pseudo-inverse fallback.")
            L_mat = np.linalg.pinv(G) @ self.liouville_matrix_
            eigenvalues, eigenvectors = np.linalg.eig(L_mat)

        # Нормируем собственные векторы по норме, определенной через G: v^* G v = 1
        for i in range(eigenvectors.shape[1]):
            v = eigenvectors[:, i]
            norm = np.sqrt(np.abs(v.conj().T @ G @ v))
            eigenvectors[:, i] = v / (norm + 1e-12) 

        # Сортировка по модулю собственных значений (от больших к меньшим)
        self.eigenvalues_, self.eigenvectors_ = sort_eigendecomposition(
            eigenvalues,
            eigenvectors,
            self.stability_policy.sorting_strategy,
        )
        
        return self

    def get_eigenfunctions(self):
        if self.eigenvalues_ is None:
            raise RuntimeError("Operator not fitted.")
        return self.eigenvalues_, self.eigenvectors_


class FractionalDMD(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            liouville_operator,
            n_quad_points=20,
            regularization=1e-8,
            regularization_policy=None,
            stability_policy=None,
            verbose=False,
    ):
        self.liouville_operator = liouville_operator
        self.okhs = liouville_operator.okhs
        self.n_quad_points = n_quad_points
        self.regularization = regularization
        self.regularization_policy = regularization_policy or RegularizationPolicy()
        self.stability_policy = stability_policy or StabilityPolicy()
        self.verbose = verbose
        
        self.modes_ = None
        self._quad_cache = None


    def _get_jacobi_rule(self):
        if self._quad_cache is None:
            self._quad_cache = roots_jacobi(self.n_quad_points, self.okhs.q - 1, 0)
        return self._quad_cache
        
    def _integrate_observable_projection(self, trajectory, observable_func):
        """
        Вычисляет <g_id, Phi_k>_OKHS.
        Это равно (T g_id)(trajectory) = C_q * int (T-t)^(q-1) g_id(traj(t)) dt.
        """
        T = self.okhs._get_trajectory_duration(trajectory)
        if T <= 1e-14: 
            return np.zeros_like(observable_func(0.0))

        nodes, weights = self._get_jacobi_rule()
        t_nodes = T * (nodes + 1) / 2
        jacobian_factor = (T / 2.0) ** self.okhs.q
        
        integral_sum = 0.0
        for k in range(self.n_quad_points):
            val = observable_func(t_nodes[k])
            integral_sum += weights[k] * val
            
        return self.okhs.C_q * jacobian_factor * integral_sum


    def compute_identity_projections(self, trajectories):
        """
        Вычисляет матрицу Y (проекции g_id на occupation kernels).
        Y_ki = < (g_id)_i, Phi_k >
        """
        n_traj = len(trajectories)
        n_features = trajectories[0].shape[1]
        Y = np.zeros((n_traj, n_features))
        
        for k in range(n_traj):
            traj = trajectories[k]
            T_traj = self.okhs._get_trajectory_duration(traj)
            
            obs_func = lambda t: self.okhs._evaluate_trajectory_at_time(traj, t, T_traj)
            Y[k, :] = self._integrate_observable_projection(traj, obs_func)
            
        return Y


    def compute_eigenfunction_projections(self, Y, V):
        """
        Переход к базису собственных функций.
        B = V^* Y
        """
        return V.conj().T @ Y


    def solve_modes(self, W, B):
        """
        Решение системы W * Xi = B.
        """
        regularization = max(self.regularization, self.regularization_policy.base_jitter)
        W_reg = W + regularization * np.eye(W.shape[0])
        
        try:
            Xi = np.linalg.solve(W_reg, B)
        except np.linalg.LinAlgError:
            Xi = np.linalg.pinv(W_reg) @ B
        return Xi


    def fit(self, trajectories=None):
        if trajectories is None:
            trajectories = self.okhs.train_trajectories_
            
        if self.liouville_operator.eigenvectors_ is None:
             raise ValueError("Liouville operator must be fitted.")
             
        V = self.liouville_operator.eigenvectors_
        G = self.okhs.gram_matrix_
        
        Y = self.compute_identity_projections(trajectories)
        
        B = self.compute_eigenfunction_projections(Y, V)
        
        # Compute W = V^* G V (Gram matrix in eigenbasis)
        W = V.conj().T @ G @ V
        
        # Solve for modes Xi
        self.modes_ = self.solve_modes(W, B)
        
        return self


    def fit_initial_coefficients(self, initial_trajectory, eig=None, Xi=None):
        """
        Определяет коэффициенты c_j из решения системы уравнений:
        
        x(t_k) = sum_j c_j * Xi_j * E_q(lambda_j * t_k^q)
        
        Используются все точки initial_trajectory. Система решается в смысле 
        наименьших квадратов.
        
        Parameters
        ----------
        initial_trajectory : array-like, shape (K, n_features)
            Начальный сегмент траектории (от t=0).
            
        Returns
        -------
        c : array, shape (n_modes,), dtype=complex
            Коэффициенты разложения.
        """
        initial_trajectory = np.asarray(initial_trajectory)
        K, n_features = initial_trajectory.shape

        if eig is None:
            eig = np.asarray(self.liouville_operator.eigenvalues_)
        if Xi is None:
            Xi = np.asarray(self.modes_)  # (n_modes, n_features)
        n_modes = len(eig)
        
        # Проверка размерности
        validate_initial_coefficient_feasibility(initial_trajectory, n_modes)
        
        # Временная сетка: t_k = k * dt, k=0..K-1
        t_grid = np.arange(K) * self.okhs.dt
        
        # Строим систему A @ c ≈ b
        # A: (K*n_features, n_modes), b: (K*n_features,)
        A = np.zeros((K * n_features, n_modes), dtype=np.complex128)
        b = initial_trajectory.reshape(K * n_features).astype(np.complex128)
        
        for k, t in enumerate(t_grid):
            ml = mittag_leffler(eig * (t ** self.okhs.q), self.okhs.q, 1)  # (n_modes,)
            
            # блок строк [k*d : (k+1)*d, :]
            # Строка для признака d=0..n_features-1: A[k*d+d, j] = ml[j] * Xi[j, d]
            A[k * n_features:(k + 1) * n_features, :] = (ml[:, None] * Xi).T
        

        # Решаем систему в смысле наименьших квадратов с регуляризацией
        n_modes = A.shape[1]
        alpha = max(self.regularization, self.regularization_policy.base_jitter)

        # Вычисляем A^H * A + alpha * I
        # .conj().T — это эрмитово сопряжение
        A_stack = A.conj().T @ A
        reg_matrix = A_stack + alpha * np.eye(n_modes)

        b_stack = A.conj().T @ b
        try:
            c = np.linalg.solve(reg_matrix, b_stack)
        except np.linalg.LinAlgError:
            c = np.linalg.pinv(reg_matrix) @ b_stack

        self.initial_coefficients_ = c
                
        return c

    def predict(self, initial_trajectory, t_span, stability_threshold=None):
        """
        Предсказание траектории на основе начального сегмента.
        
        Parameters
        ----------
        initial_trajectory : array-like, shape (K, n_features)
            Начальный сегмент траектории (используется для определения c_j).
            
        t_span : array-like, shape (n_predict,)
            Временные точки для предсказания (физическое время от t=0).
            
        Returns
        -------
        x_pred : array, shape (n_predict, n_features)
            Предсказанная траектория.
        """
        initial_trajectory = np.asarray(initial_trajectory)
        t_span = np.asarray(t_span)
        
        eig_full = np.asarray(self.liouville_operator.eigenvalues_)
        Xi_full = np.asarray(self.modes_)  # (n_modes, n_features)

        stable_mask = select_stable_modes(
            eig_full,
            stability_policy=self.stability_policy,
            stability_threshold=stability_threshold,
        )
        eig = eig_full[stable_mask]
        Xi = Xi_full[stable_mask]
        if eig.size == 0:
            raise ValueError("No stable modes remain after applying the stability policy.")

        c = self.fit_initial_coefficients(initial_trajectory, Xi=Xi, eig=eig)
        
        # Вычисляем эволюцию Mittag-Leffler для всех t и всех λ
        t_q = (t_span.astype(np.complex128) ** self.okhs.q)[:, None]  # (n_pred, 1)
        lam = eig[None, :]  # (1, n_modes)
        ML = mittag_leffler(lam * t_q, self.okhs.q, 1)  # (n_pred, n_modes)
        
        # x(t) = sum_j c_j * E_q(λ_j t^q) * Xi_j
        # В матричной форме: ML @ diag(c) @ Xi = ML @ (c[:, None] * Xi)

        X = c[:, None] * Xi  # (n_modes, n_features)
        x_pred = ML @ X      # (n_pred, n_features)
        
        return np.real(x_pred)
    

    def plot_predict(self, initial_trajectory, t_span, stability_threshold=0.05):
        """
        Deprecated wrapper around plot_forecast_diagnostics.
        """
        warnings.warn(
            "plot_predict is deprecated; use plot_forecast_diagnostics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return plot_forecast_diagnostics(
            self,
            initial_trajectory=initial_trajectory,
            t_span=t_span,
            stability_threshold=stability_threshold,
        )


def plot_forecast_diagnostics(fdmd, initial_trajectory, t_span, stability_threshold=None):
    import matplotlib.pyplot as plt
    tab10 = plt.get_cmap("tab10")
    tab20 = plt.get_cmap("tab20")

    initial_trajectory = np.asarray(initial_trajectory)
    t_span = np.asarray(t_span)

    eig_full = np.asarray(fdmd.liouville_operator.eigenvalues_)
    xi_full = np.asarray(fdmd.modes_)
    stable_mask = select_stable_modes(
        eig_full,
        stability_policy=fdmd.stability_policy,
        stability_threshold=stability_threshold,
    )
    eig = eig_full[stable_mask]
    xi = xi_full[stable_mask]
    n_modes = len(eig)

    coefficients = fdmd.fit_initial_coefficients(initial_trajectory, Xi=xi, eig=eig)
    t_q = (t_span.astype(np.complex128) ** fdmd.okhs.q)[:, None]
    lam = eig[None, :]
    mittag = mittag_leffler(lam * t_q, fdmd.okhs.q, 1)
    predicted = np.real(mittag @ (coefficients[:, None] * xi))

    abs_c = np.abs(coefficients)
    top_k = min(15, n_modes)
    top_idx = np.argsort(abs_c)[::-1][:top_k]
    other_idx = np.setdiff1d(np.arange(n_modes), top_idx)
    colors = [tab10(index) for index in range(top_k)] if top_k <= 10 else [tab20(index) for index in range(top_k)]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("Анализ компонентов модели fDMD (OKHS)", fontsize=16, fontweight="bold")

    axis_ml = axes[0, 0]
    for color_index, mode_index in enumerate(top_idx):
        axis_ml.plot(
            t_span,
            np.real(mittag[:, mode_index]),
            color=colors[color_index],
            linewidth=1.5,
            label=f"j={mode_index}  λ={eig[mode_index]:.3f}",
        )
    axis_ml.set_xlabel("Время t")
    axis_ml.set_ylabel("Re(E_q(λ t^q))")
    axis_ml.set_title(f"Mittag-Leffler функции (топ-{top_k})")
    axis_ml.legend(loc="best", fontsize=8, ncol=2)
    axis_ml.grid(True, alpha=0.3)

    axis_c = axes[0, 1]
    positions = np.arange(top_k)
    bars_c = axis_c.bar(positions, abs_c[top_idx], color=colors, edgecolor="black", linewidth=0.5)
    axis_c.set_xlabel("Индекс моды (отсортированы по |c|)")
    axis_c.set_ylabel("|c_j|")
    axis_c.set_title("Коэффициенты разложения")
    axis_c.set_xticks(positions)
    axis_c.set_xticklabels([str(index) for index in top_idx], rotation=45)
    for bar, value in zip(bars_c, abs_c[top_idx]):
        axis_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center",
                    va="bottom", fontsize=7)

    axis_xi = axes[1, 0]
    abs_xi0 = np.abs(xi[top_idx, 0])
    bars_xi = axis_xi.bar(positions, abs_xi0, color=colors, edgecolor="black", linewidth=0.5)
    axis_xi.set_xlabel("Индекс моды (отсортированы по |c|)")
    axis_xi.set_ylabel("|Xi[j,0]|")
    axis_xi.set_title("Первая компонента мод Лиувилля")
    axis_xi.set_xticks(positions)
    axis_xi.set_xticklabels([str(index) for index in top_idx], rotation=45)
    for bar, value in zip(bars_xi, abs_xi0):
        axis_xi.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center",
                     va="bottom", fontsize=7)

    axis_eig = axes[1, 1]
    if len(other_idx) > 0:
        axis_eig.scatter(np.real(eig[other_idx]), np.imag(eig[other_idx]), c="gray", alpha=0.5, s=20,
                         label="прочие моды")
    for color_index, mode_index in enumerate(top_idx):
        axis_eig.scatter(
            np.real(eig[mode_index]),
            np.imag(eig[mode_index]),
            color=colors[color_index],
            s=80,
            edgecolor="black",
            linewidth=0.5,
        )
        axis_eig.annotate(str(mode_index), (np.real(eig[mode_index]), np.imag(eig[mode_index])),
                          textcoords="offset points", xytext=(5, 5), fontsize=6)
    axis_eig.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axis_eig.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    axis_eig.set_xlabel("Re(λ)")
    axis_eig.set_ylabel("Im(λ)")
    axis_eig.set_title("Собственные значения λ_j")
    axis_eig.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if fdmd.verbose:
        unstable = np.where(np.real(eig) > 0.1)[0]
        print("\n=== Диагностика ===")
        print(f"Всего мод: {n_modes}, показано топ-{top_k} по |c|")
        print(f"Диапазон Re(λ): [{np.min(np.real(eig)):.3f}, {np.max(np.real(eig)):.3f}]")
        print(f"Диапазон Im(λ): [{np.min(np.imag(eig)):.3f}, {np.max(np.imag(eig)):.3f}]")
        if len(unstable) > 0:
            print("Моды с положительной действительной частью (возможный рост):")
            for mode_index in unstable[:5]:
                print(f"  j={mode_index}, λ={eig[mode_index]:.3f}, |c|={abs_c[mode_index]:.3f}")

    return predicted

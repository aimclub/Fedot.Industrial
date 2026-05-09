import numpy as np
from scipy.special import roots_jacobi
from sklearn.base import BaseEstimator
import torch

try:  # pragma: no cover - dependency is expected, but keep a safe fallback
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    from tqdm import tqdm  # type: ignore


from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.policies import RegularizationPolicy, StabilityPolicy
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.matrix_utils import (
    sort_eigendecomposition,
    validate_liouville_shapes,
    MatrixComputationProgress
)

TQDM_FACTORY = tqdm
TQDM_WRITE = tqdm.write


class FractionalLiouvilleOperator(BaseEstimator):
    """
    Оператор Лиувилля дробного порядка с использованием квадратур Якоби для численного интегрирования.

    Реализует конечномерное представление оператора P A_{f,q} P.
    Элементы матрицы вычисляются через однократный интеграл с сингулярным весом:

    A_{ij} = <A* mu_i, mu_j>
           = C_q * (T - \\tau)^{q-1} [K(xi_j(\\tau), xi_i(T)) - K(xi_j(\\tau), xi_i(0))] d\\tau
    """

    def __init__(
            self,
            okhs_transformer,
            ready_operator=None,
            n_quad_points=20,
            pairwise_block_size=64,
            regularization_policy=None,
            stability_policy=None,
            verbose=False,
    ):
        self.okhs = okhs_transformer
        self.n_quad_points = n_quad_points
        self.pairwise_block_size = pairwise_block_size
        self.regularization_policy = regularization_policy or RegularizationPolicy()
        self.stability_policy = stability_policy or StabilityPolicy()
        self.show_progress = getattr(okhs_transformer, "show_progress", False)
        self.progress_leave = getattr(okhs_transformer, "progress_leave", False)
        self.verbose = verbose

        if ready_operator is not None:
            self.ready_operator = torch.tensor(ready_operator, dtype=torch.float64, device=okhs_transformer.device) 
        else:
            self.ready_operator = None

        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.liouville_matrix_ = None

        self._quad_cache = None

    def _get_jacobi_rule(self):
        """Возвращает узлы и веса Якоби как тензоры."""
        if self._quad_cache is None:
            q = self.okhs.q
            nodes, weights = roots_jacobi(self.n_quad_points, q - 1, 0)
            self._quad_cache = (
                torch.tensor(nodes, dtype=torch.float64),
                torch.tensor(weights, dtype=torch.float64)
            )
        return self._quad_cache

    def _build_liouville_cache(self, trajectories):
        quadrature_cache = getattr(self.okhs, '_train_quadrature_cache_', None)
        
        # Убеждаемся, что мы пробрасываем отнормированные сетки при пересборке кэша
        if quadrature_cache is None or len(quadrature_cache['values']) != len(trajectories):
            t_grids_norm = getattr(self.okhs, 'train_t_grids_', None)
            quadrature_cache = self.okhs._build_quadrature_cache(trajectories, t_grids_norm=t_grids_norm)

        # Собираем начальные и конечные точки (уже нормализованные в _build_quadrature_cache)
        start_points = []
        end_points = []
        for trajectory in trajectories:
            normalized = self.okhs._normalize_trajectory(trajectory)
            start_points.append(normalized[0])
            end_points.append(normalized[-1])

        device = quadrature_cache['values'].device
        return {
            'quadrature': quadrature_cache,
            'start_points': torch.stack(start_points).to(device),
            'end_points': torch.stack(end_points).to(device),
        }

    def _compute_liouville_entry(self, traj_i, traj_j, t_grid_i_norm=None, t_grid_j_norm=None):
        """
        Вычисляет одиночный элемент A_{ij}.
        """
        # метод из OKHS, который сам запросит менеджер сеток
        T_i = self.okhs._get_trajectory_duration(traj_i, t_grid_norm=t_grid_i_norm)
        T_j = self.okhs._get_trajectory_duration(traj_j, t_grid_norm=t_grid_j_norm)

        if T_i <= 1e-14 or T_j <= 1e-14:
            return 0.0

        _, weights = self._get_jacobi_rule()
        
        # Пробрасываем сетку для j-й траектории
        values_j = self.okhs._evaluate_trajectory_at_nodes(traj_j, T_j, t_grid_norm=t_grid_j_norm)
        normalized_i = self.okhs._normalize_trajectory(traj_i)

        end_point_i = normalized_i[-1]
        start_point_i = normalized_i[0]
        scale_j = (T_j / 2.0) ** self.okhs.q

        device = values_j.device
        weights = weights.to(device)

        kernel_to_end = self.okhs._compute_kernel_matrix_between_samples(values_j, end_point_i.unsqueeze(0)).view(-1)
        kernel_to_start = self.okhs._compute_kernel_matrix_between_samples(
            values_j, start_point_i.unsqueeze(0)).view(-1)

        weighted_sum = torch.dot(weights, kernel_to_end - kernel_to_start).item()
        return self.okhs.C_q * scale_j * weighted_sum

    def _compute_liouville_block_from_cache(
            self,
            left_start_points,
            left_end_points,
            right_values,
            right_scales,
            weights):
        batch_kernel = getattr(self.okhs.kernel, "_compute_batch_kernel", None)
        if callable(batch_kernel):
            # right_values: (N_r, Q, d) -> (N_r, Q, 1, d)
            # left_points: (N_l, d) -> (1, 1, N_l, d)
            kernel_to_end = batch_kernel(
                right_values.unsqueeze(2),
                left_end_points.view(1, 1, len(left_end_points), -1)
            ).to(dtype=torch.float64, device=weights.device)

            kernel_to_start = batch_kernel(
                right_values.unsqueeze(2),
                left_start_points.view(1, 1, len(left_start_points), -1)
            ).to(dtype=torch.float64, device=weights.device)

            # Свертка по оси узлов квадратуры (Q)
            weighted_end = torch.einsum('p,rpl->rl', weights, kernel_to_end)
            weighted_start = torch.einsum('p,rpl->rl', weights, kernel_to_start)

            return (self.okhs.C_q * right_scales.unsqueeze(1) * (weighted_end - weighted_start)).T

        liouville_block = torch.zeros((len(left_start_points), len(right_values)),
                                      dtype=torch.float64, device=weights.device)
        for left_index in range(len(left_start_points)):
            for right_index in range(len(right_values)):
                kernel_to_end = self.okhs._compute_kernel_matrix_between_samples(
                    right_values[right_index],
                    left_end_points[left_index].unsqueeze(0),
                ).view(-1)
                kernel_to_start = self.okhs._compute_kernel_matrix_between_samples(
                    right_values[right_index],
                    left_start_points[left_index].unsqueeze(0),
                ).view(-1)

                val = torch.dot(weights, kernel_to_end - kernel_to_start)
                liouville_block[left_index, right_index] = self.okhs.C_q * right_scales[right_index] * val

        return liouville_block

    def fit(self, trajectories=None):

        if self.ready_operator is not None:
            self.liouville_matrix_ = self.ready_operator
            
            eigenvalues, eigenvectors = torch.linalg.eig(self.liouville_matrix_)
            self.eigenvalues_, self.eigenvectors_ = sort_eigendecomposition(
            eigenvalues,
            eigenvectors,
            self.stability_policy.sorting_strategy,
            )

            return self
        
        if trajectories is None:
            if not hasattr(self.okhs, 'train_trajectories_'):
                raise ValueError("OKHSTransformer must be fitted first.")
            trajectories = self.okhs.train_trajectories_

        n_traj = len(trajectories)
        
        # Кэш соберется корректно, так как в _build_liouville_cache мы добавили 
        # проброс t_grids_norm, если кэша вдруг нет.
        liouville_cache = self._build_liouville_cache(trajectories)
        values = liouville_cache['quadrature']['values']
        scales = liouville_cache['quadrature']['scales']
        weights = liouville_cache['quadrature']['weights']
        start_points = liouville_cache['start_points']
        end_points = liouville_cache['end_points']

        device = values.device
        self.liouville_matrix_ = torch.zeros((n_traj, n_traj), dtype=torch.float64, device=device)

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
            L_mat = torch.linalg.solve(G, self.liouville_matrix_)
            eigenvalues, eigenvectors = torch.linalg.eig(L_mat)
        except RuntimeError as e:
            if self.regularization_policy.fallback_solver != "pinv":
                raise
            if self.verbose:
                print(f"Generalized eig failed ({e}), using pseudo-inverse fallback.")
            L_mat = torch.linalg.pinv(G) @ self.liouville_matrix_
            eigenvalues, eigenvectors = torch.linalg.eig(L_mat)

        G_complex = G.to(eigenvectors.dtype)
        G_v = G_complex @ eigenvectors
        norm_sq = torch.sum(eigenvectors.conj() * G_v, dim=0)
        norms = torch.sqrt(torch.abs(norm_sq))
        eigenvectors = eigenvectors / (norms.unsqueeze(0) + 1e-12)

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
import torch
import torch.nn as nn
from scipy.special import gamma, roots_jacobi


def differentiable_linear_interpolation(t_grid: torch.Tensor, z_trajectory: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    """
    Линейная интерполяция латентных траекторий z(t) на произвольную сетку узлов.
    
    Пространства:
    - t_grid: R^K - базовая временная сетка наблюдений
    - z_trajectory: R^{N x K x m} - латентные представления траекторий в базисе \tilde{H}
    - t_nodes: R^{N x Q} - узлы квадратур для каждой траектории батча
    """
    N, Q = t_nodes.shape
    
    # Ограничиваем узлы пределами сетки
    t_nodes_clamped = torch.clamp(t_nodes, min=t_grid[0], max=t_grid[-1])
    
    # Поиск индексов правых границ (возвращает индексы от 1 до K-1)
    idx = torch.searchsorted(t_grid, t_nodes_clamped)
    idx = torch.clamp(idx, min=1, max=len(t_grid) - 1)
    
    t_left = t_grid[idx - 1]
    t_right = t_grid[idx]
    weights_right = (t_nodes_clamped - t_left) / (t_right - t_left + 1e-12)
    weights_left = 1.0 - weights_right
    
    # Извлечение значений латентных векторов на границах интервалов
    batch_indices = torch.arange(N, device=t_grid.device).view(-1, 1).expand(N, Q)
    
    z_left = z_trajectory[batch_indices, idx - 1, :]   # (N, Q, m)
    z_right = z_trajectory[batch_indices, idx, :]      # (N, Q, m)
    
    w_l = weights_left.unsqueeze(-1)  # (N, Q, 1)
    w_r = weights_right.unsqueeze(-1) # (N, Q, 1)
    
    return z_left * w_l + z_right * w_r


class DeepFractionalDMDLoss(nn.Module):
    """
    Вычисляет интегральную невязку (Adjoint Loss).
    """
    def __init__(self, latent_dim: int, q: float, n_quad_points: int = 20, device: str = 'cpu'):
        super().__init__()
        self.q = q
        self.latent_dim = latent_dim
        
        # Матрица оператора Лиувилля 
        self.W = nn.Parameter(torch.eye(latent_dim, dtype=torch.float64, device=device) + 
                              torch.randn(latent_dim, latent_dim, dtype=torch.float64, device=device) * 0.1)

        nodes, weights = roots_jacobi(n_quad_points, q - 1, 0)
        self.register_buffer('nodes', torch.tensor(nodes, dtype=torch.float64, device=device))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float64, device=device))
        self.c_q = 1.0 / gamma(q)

    def forward(self, t_grid: torch.Tensor, z_trajectory: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Векторизованное вычисление лосса на батче.
        
        t_grid: (K,) - моменты времени наблюдений
        z_trajectory: (N, K, m) - латентные траектории
        T: (N,) - длительности каждой траектории из батча
        """
        
        # Переход от узлов Якоби u \in [-1, 1] к сетке \tau \in [0, T_i]
        tau_nodes = (T.view(-1, 1) / 2.0) * (self.nodes.unsqueeze(0) + 1.0) # (N, Q)
        z_nodes = differentiable_linear_interpolation(t_grid, z_trajectory, tau_nodes)
        
        scale = self.c_q * ((T / 2.0) ** self.q) # (N,)
        integral_sum = torch.einsum('q,nqm->nm', self.weights, z_nodes)
        fractional_integral = scale.unsqueeze(1) * integral_sum

        w_action = torch.matmul(fractional_integral, self.W.T)
        z_start = z_trajectory[:, 0, :]
        z_end = differentiable_linear_interpolation(t_grid, z_trajectory, T.view(-1, 1)).squeeze(1)
        boundary_diff = z_end - z_start
        residual = boundary_diff - w_action
        loss_adjoint = torch.mean(torch.linalg.norm(residual, dim=1)**2)
        
        return loss_adjoint
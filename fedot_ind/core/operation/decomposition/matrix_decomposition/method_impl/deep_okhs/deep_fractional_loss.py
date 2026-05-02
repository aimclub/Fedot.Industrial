import torch
import torch.nn as nn
from scipy.special import gamma, roots_jacobi

class DeepFractionalDMDLoss(nn.Module):
    """
    Вычисляет интегральную невязку с использованием метода коллокаций 
    и кэшированием тензоров интерполяции.
    """
    def __init__(self, latent_dim: int, q: float, n_quad_points: int = 20, num_tk_samples: int = 5, device: str = 'cpu'):
        super().__init__()
        self.q = q
        self.latent_dim = latent_dim
        self.num_tk_samples = num_tk_samples
        
        # Матрица оператора Лиувилля \in \mathbb{R}^{m \times m}
        self.W = nn.Parameter(torch.eye(latent_dim, dtype=torch.float64, device=device) + 
                              torch.randn(latent_dim, latent_dim, dtype=torch.float64, device=device) * 0.1)

        nodes, weights = roots_jacobi(n_quad_points, q - 1, 0)
        self.register_buffer('nodes', torch.tensor(nodes, dtype=torch.float64, device=device))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float64, device=device))
        self.c_q = 1.0 / gamma(q)
        
        # Внутреннее состояние для кэширования графа интерполяции
        self._cache_signature = None
        self.cache = {}

    def _build_interpolation_cache(self, t_grid: torch.Tensor, T: torch.Tensor):
        """
        Прекомпиляция фиксированной сетки коллокаций и индексов интерполяции.
        Пространства:
        - t_grid: R^K
        - T: R^N
        """
        N = T.shape[0]
        device = T.device
        
        # 1. Фиксированная равномерная сетка точек коллокации вместо Монте-Карло
        # t_k \in R^{N \times num_tk_samples}
        steps = torch.linspace(1.0 / self.num_tk_samples, 1.0, self.num_tk_samples, device=device)
        t_k = steps.unsqueeze(0) * T.view(-1, 1)

        # 2. Узлы квадратур Гаусса-Якоби
        # tau_nodes \in R^{N \times num_tk_samples \times Q}
        tau_nodes = (t_k.unsqueeze(-1) / 2.0) * (self.nodes.view(1, 1, -1) + 1.0)
        
        # 3. Масштабирующий множитель дробного интеграла
        self.cache['scale'] = self.c_q * ((t_k / 2.0) ** self.q)  # (N, num_tk_samples)

        # Функция для расчета индексов и весов отрезков (выполняется 1 раз)
        def precompute_weights(nodes: torch.Tensor):
            flat_nodes = torch.clamp(nodes.view(N, -1), min=t_grid[0], max=t_grid[-1])
            idx = torch.searchsorted(t_grid, flat_nodes)
            idx = torch.clamp(idx, min=1, max=len(t_grid) - 1)
            
            t_l = t_grid[idx - 1]
            t_r = t_grid[idx]
            w_r = (flat_nodes - t_l) / (t_r - t_l + 1e-12)
            w_l = 1.0 - w_r
            
            batch_idx = torch.arange(N, device=device).view(-1, 1).expand_as(idx)
            return batch_idx, idx, w_l.unsqueeze(-1), w_r.unsqueeze(-1)

        # Кэшируем индексы для узлов интегрирования \tau
        b_idx_tau, idx_tau, wl_tau, wr_tau = precompute_weights(tau_nodes)
        self.cache['tau'] = (b_idx_tau, idx_tau, wl_tau, wr_tau)
        
        # Кэшируем индексы для границ интервалов t_k (для дифференциальной невязки)
        b_idx_tk, idx_tk, wl_tk, wr_tk = precompute_weights(t_k)
        self.cache['tk'] = (b_idx_tk, idx_tk, wl_tk, wr_tk)
        
        # Сохраняем сигнатуру батча, чтобы понимать, когда кэш "протух"
        self._cache_signature = (N, float(T[0].item()), float(t_grid[-1].item()))

    def _fast_interpolate(self, z_trajectory: torch.Tensor, cache_key: str, original_shape: tuple) -> torch.Tensor:
        """
        Векторизованная сборка тензоров O(1) без поиска по сетке.
        """
        b_idx, idx, wl, wr = self.cache[cache_key]
        
        z_left = z_trajectory[b_idx, idx - 1, :]
        z_right = z_trajectory[b_idx, idx, :]
        z_interp = z_left * wl + z_right * wr
        
        return z_interp.view(original_shape)

    def forward(self, t_grid: torch.Tensor, z_trajectory: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        N = z_trajectory.shape[0]
        Q = len(self.nodes)
        
        # Проверяем инвалидацию кэша (например, если пришел неполный батч в конце эпохи)
        current_sig = (N, float(T[0].item()), float(t_grid[-1].item()))
        if self._cache_signature != current_sig:
            self._build_interpolation_cache(t_grid, T)

        # --- Аналитическая фаза графа ---
        
        # 1. Сборка интерполированных значений в узлах \tau (N, S, Q, m)
        z_nodes = self._fast_interpolate(
            z_trajectory, 'tau', 
            original_shape=(N, self.num_tk_samples, Q, self.latent_dim)
        )

        # 2. Вычисление дробного интеграла Римана-Лиувилля
        integral_sum = torch.einsum('q, nkqm -> nkm', self.weights, z_nodes)
        fractional_integral = self.cache['scale'].unsqueeze(-1) * integral_sum
        w_action = torch.matmul(fractional_integral, self.W.T)
        
        # 3. Дифференциальная невязка на концах [0, t_k]
        z_start = z_trajectory[:, 0, :].unsqueeze(1) # (N, 1, m)
        z_tk = self._fast_interpolate(
            z_trajectory, 'tk', 
            original_shape=(N, self.num_tk_samples, self.latent_dim)
        )
        boundary_diff = z_tk - z_start 
        
        # 4. Лосс
        residual = boundary_diff - w_action
        loss_adjoint = torch.mean(torch.linalg.norm(residual, dim=-1)**2)
        
        return loss_adjoint
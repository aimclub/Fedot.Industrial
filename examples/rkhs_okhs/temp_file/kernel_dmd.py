"""
Простая реализация классического Kernel DMD для сравнения с OKHS. 
Этот код не оптимизирован и предназначен только для демонстрационных целей.
"""


import numpy as np
import torch
from scipy.linalg import pinv, eig

class ClassicalKernelDMD:
    def __init__(self, kernel, dt, rank=100, device='cpu'):
        self.kernel = kernel
        self.dt = dt
        self.rank = rank
        self.device = device
        
        self.X_basis = None          # Опорные снапшоты в \mathbb{R}^{r \times n}
        self.eigenvalues_cont = None # Непрерывные собственные числа \lambda_k \in \mathbb{C}
        self.W = None                # Правые собственные векторы матрицы Купмана A
        self.W_inv = None
        self.Xi_W = None             # Моды Лиувилля \xi_k, совмещенные с вектором W

    def fit(self, trajectories):
        X_snaps, Y_snaps = [], []
        for traj in trajectories:
            X_snaps.append(traj[:-1])
            Y_snaps.append(traj[1:])
        
        X = np.vstack(X_snaps)  
        Y = np.vstack(Y_snaps)  
        M, dim = X.shape
        
        actual_rank = min(self.rank, M)
        rng = np.random.default_rng(42)
        indices = rng.choice(M, actual_rank, replace=False)
        self.X_basis = X[indices]
        
        X_basis_t = torch.tensor(self.X_basis, dtype=torch.float64, device=self.device).unsqueeze(1)
        X_t = torch.tensor(X, dtype=torch.float64, device=self.device).unsqueeze(0)
        Y_t = torch.tensor(Y, dtype=torch.float64, device=self.device).unsqueeze(0)
        
        # Вычисление прямоугольных матриц Грама
        G_X = self.kernel._compute_batch_kernel(X_basis_t, X_t).cpu().numpy()
        G_Y = self.kernel._compute_batch_kernel(X_basis_t, Y_t).cpu().numpy()
        
        G_X_pinv = pinv(G_X)          
        A = G_Y @ G_X_pinv            
        
        mu, W = eig(A)
        
        # Комплексный логарифм для перехода к непрерывному генератору Лиувилля
        eigenvalues_cont = np.log(mu + 1e-16) / self.dt
        
        # --- ФИЛЬТРАЦИЯ НЕСТАБИЛЬНЫХ МОД ---
        # Оставляем моды с вещественной частью <= 0 (с небольшим допуском на машинный ноль)
        stable_mask = eigenvalues_cont.real <= 1e-7
        
        self.eigenvalues_cont = eigenvalues_cont[stable_mask]
        self.W = W[:, stable_mask]
        self.W_inv = pinv(self.W)
        
        Xi = X.T @ G_X_pinv           
        self.Xi_W = Xi @ self.W       

        # Опционально: вывод количества отброшенных фиктивных мод
        n_filtered = len(mu) - sum(stable_mask)
        if n_filtered > 0:
            print(f"[Kernel DMD] Отфильтровано нестабильных мод: {n_filtered} из {len(mu)}")

    def predict(self, initial_segment, time_grid, return_tensor=False):
        initial_len = len(initial_segment)
        n_steps = len(time_grid)
        dim = initial_segment.shape[1]
        
        # Массив прогноза инициализируем известной предысторией
        predictions = np.zeros((n_steps, dim))
        predictions[:initial_len] = initial_segment
        
        # x_0 \in M — финальное состояние, от которого эволюционирует марковская система
        x0 = initial_segment[-1]
        t0 = time_grid[initial_len - 1]
        
        x0_t = torch.tensor(x0, dtype=torch.float64, device=self.device).view(1, 1, -1)
        X_basis_t = torch.tensor(self.X_basis, dtype=torch.float64, device=self.device).unsqueeze(1)
        k_x0 = self.kernel._compute_batch_kernel(X_basis_t, x0_t).cpu().numpy().flatten()
        
        c = self.W_inv @ k_x0
        
        # Расчет аналитического продолжения для t > t_0
        for i in range(initial_len, n_steps):
            dt_t = float(time_grid[i] - t0)
            time_dynamics = np.exp(self.eigenvalues_cont * dt_t) * c
            predictions[i] = (self.Xi_W @ time_dynamics).real
            
        if return_tensor:
            return torch.tensor(predictions, dtype=torch.float64, device=self.device)
        return predictions

def build_classical_kdmd(basis_trajectories=None, time_grid=None, config=None, device=None):
    from example_common import RBFKernel
    dt = float(time_grid[1] - time_grid[0])
    
    # Используем базовое ядро RBF (по аналогии со стандартным базисом OKHS)
    kernel = RBFKernel(gamma=2.0)
    kdmd = ClassicalKernelDMD(kernel=kernel, dt=dt, rank=1000, device=device)
    kdmd.fit(basis_trajectories)
    return kdmd
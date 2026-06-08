import torch

class TimeGridManager:
    """
    Абстракция для работы с физическим и нормированным временем.
    Приводит все траектории к глобальному масштабу T_norm для численной 
    устойчивости вычислений дробных интегралов и функции Миттаг-Леффлера.
    """
    def __init__(self, normalization_time: float = None, dt: float = 1.0):
        self.T_norm = normalization_time
        self.dt = dt

        # Отнормированная и исходная временная сетка.
        self.train_t_grids_norm_ = None
        self.phys_grids = None
       
    def fit(self, trajectories, t_grids=None):
        """
        Определяет глобальный масштаб времени T_norm и сохраняет 
        отнормированные обучающие сетки.
        """
        device = trajectories[0].device if isinstance(trajectories[0], torch.Tensor) else 'cpu'
        
        if t_grids is not None:
            self.phys_grids = [torch.as_tensor(t, dtype=torch.float64, device=device) for t in t_grids]
        else:
            self.phys_grids = [self.build_default_grid(len(traj), device) for traj in trajectories]

        if self.T_norm is None:
            self.T_norm = max(t[-1].item() for t in self.phys_grids if len(t) > 0)
            
        self.train_t_grids_norm_ = [self.normalize(t) for t in self.phys_grids]
        
        return self

    def normalize(self, t: torch.Tensor) -> torch.Tensor:
        """Переводит физическое время t в нормированное tau."""
        if self.T_norm is None:
            raise ValueError("TimeGridManager must be fitted before normalization.")
        return t / self.T_norm

    def denormalize(self, tau: torch.Tensor) -> torch.Tensor:
        """Переводит нормированное время tau в физическое t."""
        if self.T_norm is None:
            raise ValueError("TimeGridManager must be fitted before denormalization.")
        return tau * self.T_norm

    def build_default_grid(self, length: int, device='cpu') -> torch.Tensor:
        """Генерирует равномерную сетку физического времени для обратной совместимости."""
        return torch.arange(length, dtype=torch.float64, device=device) * self.dt

    def interpolate(self, t_grid: torch.Tensor, x: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
        """
        Векторизованная линейная интерполяция значений состояния x.
        Все тензоры времени (t_grid и target_t) должны быть в одной шкале (например, нормированной).
        
        Args:
            t_grid: (K,) монотонная временная сетка наблюдений.
            x: (K, d) значения траектории.
            target_t: (N,) узлы, в которых нужно вычислить значения.
        Returns:
            (N, d) интерполированные значения.
        """
        t_grid = t_grid.to(target_t.device)

        target_t = torch.clamp(target_t, min=t_grid[0], max=t_grid[-1])
        
        # Индексы правых границ интервалов
        idx = torch.searchsorted(t_grid, target_t)
        idx = torch.clamp(idx, 1, len(t_grid) - 1)
        
        t_left = t_grid[idx - 1]
        t_right = t_grid[idx]
        
        # Линейные веса для выпуклой комбинации
        w_right = (target_t - t_left) / (t_right - t_left + 1e-12)
        w_left = 1.0 - w_right
        
        x_left = x[idx - 1]
        x_right = x[idx]
        
        return x_left * w_left.unsqueeze(-1) + x_right * w_right.unsqueeze(-1)

    def get_physical_grid(self, x: torch.Tensor, t_grid=None) -> torch.Tensor:
        """
        Возвращает физическую временную сетку.
        Если t_grid не передан, генерирует равномерную сетку на основе dt.
        """
        if t_grid is not None:
            return torch.as_tensor(t_grid, dtype=torch.float64, device=x.device)
        return torch.arange(len(x), dtype=torch.float64, device=x.device) * self.dt
    
    def interpolate_batch(self, x_batch: torch.Tensor, t_grids_norm: torch.Tensor, target_taus: torch.Tensor) -> torch.Tensor:
        """
        Векторизованная батчевая интерполяция для нейросетевого пайплайна.
        Поддерживает уникальные сетки времени для каждой траектории в батче.
        """

        t_grids_norm = t_grids_norm.to(target_taus.device)

        # Если передана общая 1D сетка, расширяем ее на весь батч
        if t_grids_norm.ndim == 1:
            t_grids_norm = t_grids_norm.unsqueeze(0).expand(x_batch.shape[0], -1)

        N, K, d = x_batch.shape
        target_shape = target_taus.shape
        
        # Вытягиваем все целевые узлы в 2D (N, M), чтобы использовать searchsorted
        target_flat = target_taus.view(N, -1)
        
        # Ограничиваем экстраполяцию краями исходных сеток
        t_min = t_grids_norm[:, 0:1]
        t_max = t_grids_norm[:, -1:]
        target_flat = torch.clamp(target_flat, min=t_min, max=t_max)
        
        # Поиск индексов интервалов для всего батча одновременно
        idx = torch.searchsorted(t_grids_norm, target_flat)                                                                                                              
        idx = torch.clamp(idx, 1, K - 1)
        
        # Извлекаем левые и правые границы времени (N, M)
        t_left = torch.gather(t_grids_norm, 1, idx - 1)
        t_right = torch.gather(t_grids_norm, 1, idx)
        
        # Вычисляем веса для линейной комбинации
        w_right = (target_flat - t_left) / (t_right - t_left + 1e-12)
        w_left = 1.0 - w_right
        
        # Расширяем индексы для выборки по размерности признаков d
        # idx_expanded: (N, M, d)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, d)
        
        # Извлекаем значения состояний (N, M, d)
        x_left = torch.gather(x_batch, 1, idx_expanded - 1)
        x_right = torch.gather(x_batch, 1, idx_expanded)
        
        # Интерполируем
        x_interp = x_left * w_left.unsqueeze(-1) + x_right * w_right.unsqueeze(-1)

        return x_interp.view(*target_shape, d)
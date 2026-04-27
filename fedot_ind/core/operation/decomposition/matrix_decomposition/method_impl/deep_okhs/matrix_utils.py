from dataclasses import dataclass
import torch
import time
from tqdm import tqdm

TQDM_FACTORY = tqdm
TQDM_WRITE = tqdm.write


def _sync_perf_counter() -> float:
    """
    Вычисляет время с принудительной синхронизацией CUDA.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def validate_square_matrix_shape(matrix, label):
    """
    Проверяет, является ли входная матрица квадратной. Если нет, вызывает ValueError.
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"{label} must be square, got shape {matrix.shape}.")


def validate_liouville_shapes(gram_matrix, liouville_matrix):
    """
    Проверяет, что матрица Грама и матрица Лиувилля являются квадратными и имеют одинаковые размеры.
    """
    validate_square_matrix_shape(gram_matrix, "Gram matrix")
    validate_square_matrix_shape(liouville_matrix, "Liouville matrix")
    if gram_matrix.shape != liouville_matrix.shape:
        raise ValueError(
            "Gram and Liouville matrices must have identical shapes, "
            f"got {gram_matrix.shape} and {liouville_matrix.shape}."
        )


def validate_initial_coefficient_feasibility(initial_trajectory, n_modes):
    """
    Проверяет, достаточно ли данных в начальной траектории для определения заданного числа мод.
    """
    # Используем as_tensor, чтобы избежать лишнего копирования, если на вход уже подан тензор
    initial_trajectory = torch.as_tensor(initial_trajectory)
    if initial_trajectory.ndim != 2:
        raise ValueError("initial_trajectory must have shape (K, n_features).")

    n_points, n_features = initial_trajectory.shape
    available_equations = n_points * n_features
    if available_equations < n_modes:
        raise ValueError(
            f"Insufficient data for coefficient fit: {available_equations} equations < {n_modes} modes."
        )


def select_stable_modes(eigenvalues, stability_policy, stability_threshold=None):
    """
    Выбирает "стабильные" моды на основе политики стабильности.
    """
    if not stability_policy.drop_positive_real_modes:
        # Создаем тензор на том же устройстве, что и eigenvalues, чтобы избежать конфликтов
        return torch.ones(len(eigenvalues), dtype=torch.bool, device=eigenvalues.device)

    threshold = (
        stability_policy.threshold
        if stability_threshold is None
        else stability_threshold
    )
    # У комплексных тензоров PyTorch есть свойство .real
    return eigenvalues.real < threshold


def sort_eigendecomposition(eigenvalues, eigenvectors, sorting_strategy):
    """
    Сортирует собственные значения и собственные векторы согласно заданной стратегии.
    """
    if sorting_strategy == "real_desc":
        # Используем descending=True для сортировки по убыванию на уровне C++/CUDA
        order = torch.argsort(eigenvalues.real, descending=True)
    else:
        # У комплексных тензоров метод .abs() возвращает модуль
        order = torch.argsort(eigenvalues.abs(), descending=True)

    return eigenvalues[order], eigenvectors[:, order]


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
        self.started_at = _sync_perf_counter()
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
        elapsed_seconds = max(_sync_perf_counter() - self.started_at, 1e-9)
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
        elapsed_seconds = max(_sync_perf_counter() - self.started_at, 0.0)
        blocks_per_second = self.completed_blocks / max(elapsed_seconds, 1e-9) if self.completed_blocks else 0.0
        if self.enabled:
            TQDM_WRITE(
                f"[okhs] finished {self.matrix_name} matrix "
                f"elapsed_s={elapsed_seconds:.2f} blocks={self.completed_blocks} blk_s={blocks_per_second:.2f}"
            )
        if self.enabled and self.bar is not None and hasattr(self.bar, "close"):
            self.bar.close()

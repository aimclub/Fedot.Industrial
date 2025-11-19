import numpy as np
from sklearn.base import TransformerMixin

# from ..core.kernels import OccupationKernel
from ....transformation.representation.kernel.kernels import OccupationKernel

class OKHSTransformer(TransformerMixin):
    """Трансформер для OKHS признаков"""

    def __init__(self, q=0.7, n_components=None, eigenvalue_threshold=1e-6):
        self.q = q
        self.n_components = n_components
        self.eigenvalue_threshold = eigenvalue_threshold
        self.kernel = OccupationKernel(q=q)

    def fit(self, trajectories, y=None):
        """Обучение трансформера"""
        self.gram_matrix_ = self.kernel.compute_gram_matrix(trajectories)
        self.trajectories_ = trajectories

        return self

    def transform(self, trajectories):
        """Преобразование траекторий в OKHS пространство"""
        # Вычисление проекций на собственные векторы
        n_train = len(self.trajectories_)
        n_test = len(trajectories)

        # Матрица ядер между train и test
        kernel_matrix = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                kernel_matrix[i, j] = self.kernel._compute_trajectory_kernel(
                    trajectories[i], self.trajectories_[j]
                )

    
        # Решаем систему: W * c = kernel_matrix^T
        # где c — координаты в базисе occupation kernels
        c = np.linalg.solve(self.gram_matrix_, kernel_matrix.T).T
        
        return c  # размер (n_test, n_train)

    def fit_transform(self, trajectories):
        self.fit(trajectories)
        return self.transform(trajectories)
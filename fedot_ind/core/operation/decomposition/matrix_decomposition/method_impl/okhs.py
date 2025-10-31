import numpy as np
from sklearn.base import TransformerMixin

from ..core.kernels import OccupationKernel


class OKHSTransformer(TransformerMixin):
    """Трансформер для OKHS признаков"""

    def __init__(self, q=0.7, n_components=None):
        self.q = q
        self.n_components = n_components
        self.kernel = OccupationKernel(q=q)

    def fit(self, trajectories, y=None):
        """Обучение трансформера"""
        self.gram_matrix_ = self.kernel.compute_gram_matrix(trajectories)

        # Собственное разложение для уменьшения размерности
        eigenvalues, eigenvectors = np.linalg.eigh(self.gram_matrix_)

        # Сортировка по убыванию собственных значений
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]

        if self.n_components is not None:
            self.eigenvectors_ = self.eigenvectors_[:, :self.n_components]
            self.eigenvalues_ = self.eigenvalues_[:self.n_components]

        return self

    def transform(self, trajectories):
        """Преобразование траекторий в OKHS пространство"""
        # Вычисление проекций на собственные векторы
        n_train = self.eigenvectors_.shape[0]
        n_test = len(trajectories)

        # Матрица ядер между train и test
        kernel_matrix = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                kernel_matrix[i, j] = self.kernel._compute_trajectory_kernel(
                    trajectories[i], self.trajectories_[j]
                )

        # Проекция в OKHS пространство
        okhs_features = kernel_matrix @ self.eigenvectors_
        return okhs_features / np.sqrt(self.eigenvalues_)

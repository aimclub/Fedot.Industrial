import numpy as np

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs import OKHSTransformer
from ..utils.special_functions import mittag_leffler


class FractionalDMD:
    """Дробный Dynamic Mode Decomposition"""

    def __init__(self, q=0.7, n_modes=None):
        self.q = q
        self.n_modes = n_modes
        self.okhs = OKHSTransformer(q=q)

    def fit(self, trajectories):
        """Обучение DMD модели"""
        # Преобразование в OKHS пространство
        X_okhs = self.okhs.fit_transform(trajectories[:-1])
        Y_okhs = self.okhs.transform(trajectories[1:])

        # Оператор эволюции
        self.A_ = Y_okhs @ np.linalg.pinv(X_okhs)

        # Собственные значения и векторы
        self.eigenvalues_, self.eigenvectors_ = np.linalg.eig(self.A_)

        if self.n_modes is not None:
            idx = np.argsort(np.abs(self.eigenvalues_))[::-1][:self.n_modes]
            self.eigenvalues_ = self.eigenvalues_[idx]
            self.eigenvectors_ = self.eigenvectors_[:, idx]

        return self

    def predict(self, initial_condition, time_steps):
        """Прогноз с использованием функции Миттаг-Леффлера"""
        predictions = []
        current_state = initial_condition

        for t in time_steps:
            state_pred = np.zeros_like(current_state)

            for i, (eval, evec) in enumerate(zip(self.eigenvalues_, self.eigenvectors_.T)):
                # Использование функции Миттаг-Леффлера вместо экспоненты
                ml_value = mittag_leffler(eval * t ** self.q, self.q)
                state_pred += evec * ml_value * np.dot(evec, current_state)

            predictions.append(state_pred.copy())
            current_state = state_pred

        return np.array(predictions)

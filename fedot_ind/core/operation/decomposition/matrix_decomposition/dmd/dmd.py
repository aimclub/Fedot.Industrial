import numpy as np

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs import OKHSTransformer
from ....transformation.representation.kernel.utils import mittag_leffler


class FractionalDMD:
    """Дробный Dynamic Mode Decomposition"""

    def __init__(self, q=0.7, n_modes=None):
        self.q = q
        self.n_modes = n_modes
        self.okhs = OKHSTransformer(q=q)
    
    def fit(self, trajectories):
        """Обучение DMD модели"""
        # Преобразование в OKHS пространство
        # X_okhs = self.okhs.fit_transform(trajectories[:-1])
        self.okhs.fit(trajectories)
        X_okhs = self.okhs.transform(trajectories[:-1]) 
        Y_okhs = self.okhs.transform(trajectories[1:])

        # Оператор эволюции находится из задачи минимизации ошибки переходе к состоянию на один момент времени дальше
        self.A_ = np.linalg.pinv(X_okhs) @ Y_okhs

        # print("X", X_okhs)
        # print("Y", Y_okhs)
        # # self.A_ = Y_okhs @ np.transpose(X_okhs) @ np.linalg.inv(X_okhs @ np.transpose(X_okhs) + 1e-6 * np.eye(X_okhs.shape[0]))
        
        # Собственные значения и векторы
        self.eigenvalues_, self.eigenvectors_ = np.linalg.eig(self.A_)

        if self.n_modes is not None:
            idx = np.argsort(np.abs(self.eigenvalues_))[::-1][:self.n_modes]
            self.eigenvalues_ = self.eigenvalues_[idx]
            self.eigenvectors_ = self.eigenvectors_[:, idx]

        # Вычисление Fractional Liouville Modes
        self._compute_modes(trajectories, X_okhs)
        
        # Сохраняем обучающие данные для вычисления phi_i(x0)
        self.training_trajectories_ = trajectories

        return self

    # def predict(self, initial_condition, time_steps):
    #     """Прогноз с использованием функции Миттаг-Леффлера"""
    #     predictions = []
    #     current_state = initial_condition

    #     for t in time_steps:
    #         state_pred = np.zeros_like(current_state)

    #         for i, (eval, evec) in enumerate(zip(self.eigenvalues_, self.eigenvectors_.T)):
    #             # Использование функции Миттаг-Леффлера вместо экспоненты
    #             ml_value = mittag_leffler(eval * t ** self.q, self.q)
    #             state_pred += evec * ml_value * np.dot(evec, current_state)

    #         predictions.append(state_pred.copy())
    #         current_state = state_pred

    #     return np.array(predictions)

    def _compute_modes(self, trajectories, X_okhs):
        '''Вычисление Fractional Liouville modes'''
        n_modes = len(self.eigenvalues_)
        data_dim = trajectories[0].shape[-1] if trajectories[0].ndim > 1 else 1
        
        self.modes_ = np.zeros((n_modes, data_dim), dtype=complex)
        
        for mode_idx in range(n_modes):
            eigvec = self.eigenvectors_[:, mode_idx]
            mode_projection = np.zeros(data_dim, dtype=complex)
            
            for traj_idx, traj in enumerate(trajectories[:-1]):
                if traj.ndim == 1:
                    traj_mean = np.mean(traj)
                else:
                    traj_mean = np.mean(traj, axis=0)
                
                mode_projection += eigvec[traj_idx] * traj_mean
            
            self.modes_[mode_idx] = mode_projection


    def predict(self, initial_condition, time_steps):
        '''
        Предсказание временного ряда
        
        '''
        # Преобразуем initial_condition в траекторию
        if np.isscalar(initial_condition):
            initial_trajectory = np.array([initial_condition] * 5)
        else:
            initial_trajectory = np.array(initial_condition)
        
        # Определяем моменты времени
        if np.isscalar(time_steps):
            times = np.arange(1, time_steps + 1)
        else:
            times = np.array(time_steps)
        
        data_dim = self.modes_.shape[1]
        predictions = np.zeros((len(times), data_dim))
        
        # Вычисляем phi_i(initial_trajectory) для всех мод
        phi_values = self._compute_phi_values(initial_trajectory)
        
        # Подсчет суммы - итогового предсказания
        for t_idx, t in enumerate(times):
            prediction = np.zeros(data_dim, dtype=complex)
            
            for mode_idx in range(len(self.eigenvalues_)):
                lambda_i = self.eigenvalues_[mode_idx]
                psi_i = self.modes_[mode_idx]
                phi_i = phi_values[mode_idx]
                
                ml_value = mittag_leffler(lambda_i * (t ** self.q), self.q)
                
                prediction += psi_i * phi_i * ml_value
            
            # Берем вещественную часть
            predictions[t_idx] = np.real(prediction)
        
        # Если одномерный ряд, возвращаем плоский массив
        if data_dim == 1:
            return predictions.flatten()
        
        return predictions
    
    def _compute_phi_values(self, trajectory):
        '''
        Вычисление значений собственных функций φ_i на траектории
        '''
        n_modes = len(self.eigenvalues_)
        phi_values = np.zeros(n_modes, dtype=complex)
        
        for mode_idx in range(n_modes):
            eigvec = self.eigenvectors_[:, mode_idx]
            phi_i = 0
            
            for traj_idx, train_traj in enumerate(self.training_trajectories_[:-1]):
                
                # вычисление kernel между траекториями
                kernel_value = self.okhs.kernel._compute_trajectory_kernel(
                    train_traj, trajectory
                )
                phi_i += eigvec[traj_idx] * kernel_value
            
            phi_values[mode_idx] = phi_i
        
        return phi_values
import numpy as np
import torch
import torch.nn as nn


class DMDForecaster:
    """
    Правильная реализация DMD для многомерного прогнозирования
    """

    def __init__(self, forecast_horizon=10, n_modes=None, use_koopman=True,
                 learning_rate=0.01, epochs=1000, device='auto'):
        self.forecast_horizon = forecast_horizon
        self.n_modes = n_modes
        self.use_koopman = use_koopman  # True для Koopman, False для классического DMD
        self.learning_rate = learning_rate
        self.max_epochs = epochs
        self.device = device

    def _create_dmd_data_matrix(self, trajectories):
        """
        Создание матриц данных для DMD: X и Y
        где Y = X сдвинутый на 1 шаг вперед
        """
        X_list, Y_list = [], []
        for trajectory in trajectories:
            # trajectory: [x₁, x₂, ..., x_window_size] ∈ ℝ^(window_size)
            if len(trajectory) >= self.forecast_horizon + 1:
                # Вход: текущее окно (весь window_size точек)
                x_input = trajectory[:-self.forecast_horizon]

                # Цель: следующие forecast_horizon точек (скалярных значений!)
                y_target = trajectory[-self.forecast_horizon:]

                X_list.append(x_input)
                Y_list.append(y_target)
        # for trajectory in trajectories:
        #     # trajectory: [x₁, x₂, ..., x_T] ∈ ℝ^(T × dim)
        #     if len(trajectory) > self.forecast_horizon + 1:
        #         for i in range(len(trajectory) - self.forecast_horizon - 1):
        #             # Вход: текущее состояние
        #             x = trajectory[i:i + self.forecast_horizon]
        #             # Цель: следующие forecast_horizon состояний
        #             y = trajectory[i + 1:i + 1 + self.forecast_horizon]
        #
        #             X_list.append(x)
        #             Y_list.append(y)

        # X ∈ ℝ^(n_samples × input_dim)
        # Y ∈ ℝ^(n_samples × forecast_horizon × input_dim) для многомерного выхода
        X_torch = torch.tensor(np.array(X_list), dtype=torch.float32, device=self.device)
        Y_torch = torch.tensor(np.array(Y_list), dtype=torch.float32, device=self.device)
        return X_torch, Y_torch

    def _compute_classical_dmd(self, X, Y):
        """
        Классический DMD через SVD разложение
        A = Y X⁺, где X⁺ - псевдообратная матрица
        """

        # Вычисляем псевдообратную через SVD
        U, S, Vt = torch.svd(X)

        # Обрезаем по числу мод если задано
        if self.n_modes is not None:
            U = U[:, :self.n_modes]
            S = S[:self.n_modes]
            Vt = Vt[:self.n_modes, :]

        # Псевдообратная: X⁺ = V Σ⁺ Uᵀ
        S_inv = torch.diag(1.0 / S)
        X_pinv = Vt.T @ S_inv @ U.T

        # Оператор эволюции: A = Y X⁺
        A = Y @ X_pinv

        return A, U, S, Vt

    def _setup_koopman_model(self, input_dim):
        """
        Настройка модели на основе оператора Копмана
        В Koopman DMD мы ищем оператор в пространстве наблюдаемых
        """
        if self.n_modes is None:
            self.n_modes = min(64, input_dim * 2)  # Эвристика

        # Оператор Копмана: K ∈ ℝ^(n_modes × n_modes)
        self.K = nn.Parameter(torch.randn(self.n_modes, self.n_modes, device=self.device) * 0.01).to(self.device)

        # Энкодер: ℝ^(input_dim) → ℝ^(n_modes)
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.Tanh(),
                                     nn.Linear(512, self.n_modes)).to(self.device)

        # Декодер: ℝ^(n_modes) → ℝ^(input_dim)
        self.decoder = nn.Sequential(nn.Linear(self.n_modes, 256),
                                     nn.Tanh(), nn.Linear(256, self.forecast_horizon)).to(self.device)

    def _koopman_forward(self, x, steps=1):
        """
        Пропуск через Koopman модель
        x: ℝ^(batch_size × input_dim)
        returns: ℝ^(batch_size × steps × input_dim)
        """

        # Кодируем в пространство Купмана
        z = self.encoder(x)  # ℝ^(batch_size × n_modes)

        # Применяем оператор Копмана ОДИН раз
        z_evolved = z @ self.K.T  # ℝ^(batch_size × n_modes)

        # Декодируем в прогноз на forecast_horizon точек
        predictions = self.decoder(z_evolved)  # ℝ^(batch_size × forecast_horizon)

        return predictions

    def fit(self, trajectory_matrix, window_size=20):
        """Обучение DMD модели"""
        self.window_size_ = window_size

        # Создаем матрицы данных для DMD
        X, Y = self._create_dmd_data_matrix(trajectory_matrix)
        self.input_dim_ = X.shape[1] if X.ndim > 1 else 1
        return self._fit_koopman_dmd(X, Y) if self.use_koopman else self._fit_classical_dmd(X, Y)

    def _fit_classical_dmd(self, X, Y):
        """Обучение классического DMD"""
        self.A, self.U, self.S, self.Vt = self._compute_classical_dmd(X, Y)

        print(f"Classical DMD: A {self.A.shape}")
        return self

    def _fit_koopman_dmd(self, X, Y):
        """Обучение Koopman DMD через оптимизацию"""
        self._setup_koopman_model(self.input_dim_)

        # Y имеет форму (n_samples, forecast_horizon, input_dim)
        # Нам нужно предсказать все шаги сразу
        dmd_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + [self.K]
        optimizer = torch.optim.Adam(dmd_params, lr=self.learning_rate)

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()

            # Предсказание на все шаги
            predictions = self._koopman_forward(X, steps=self.forecast_horizon)
            # Сравниваем со всеми целевыми шагами
            loss = nn.MSELoss()(predictions, Y)

            # Регуляризация для стабильности оператора Копмана
            reg_loss = torch.norm(self.K, p='fro') * 0.001
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {total_loss.item():.4f}')

        return self

    def predict(self, last_trajectory):
        """Прогнозирование с правильной DMD"""

        X_list = [last_trajectory[i:i + self.forecast_horizon]
                  for i in range(len(last_trajectory) - self.forecast_horizon - 1)]

        X_list = last_trajectory[-self.forecast_horizon:]
        last_trajectory_tensor = torch.tensor(np.array(X_list), dtype=torch.float32,
                                              device=self.device)  # batch_size x forecast_horizon
        if self.use_koopman:
            return self._predict_koopman(last_trajectory_tensor)
        else:
            return self._predict_classical(last_trajectory)

    def _predict_classical(self, last_window):
        """Прогнозирование классическим DMD"""
        # last_window: [x_{t-k}, ..., x_t]
        current_state = last_window[-1]  # Берем последнее состояние

        if current_state.ndim == 0:
            current_state = np.array([current_state])

        current_torch = torch.tensor(current_state, dtype=torch.float32, device=self.device)

        predictions = []

        for step in range(self.forecast_horizon):
            # Применяем оператор эволюции: x_{t+1} = A x_t
            next_state = self.A @ current_torch
            predictions.append(next_state.detach().cpu().numpy())

            # Для рекурсивного прогноза обновляем состояние
            current_torch = next_state

        return np.array(predictions)

    def _predict_koopman(self, last_trajectory_tensor):
        """Прогнозирование Koopman DMD"""
        with torch.no_grad():
            predictions = self._koopman_forward(last_trajectory_tensor, steps=self.forecast_horizon)
        return predictions.squeeze(0).cpu().numpy()

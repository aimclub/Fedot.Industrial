import numpy as np
import torch
import torch.nn as nn


class DMDForecaster:
    """
    DMD forecaster for univariate and multivariate trajectory windows.
    """

    def __init__(self, forecast_horizon=10, n_modes=None, use_koopman=True,
                 learning_rate=0.01, epochs=1000, device='auto'):
        self.forecast_horizon = forecast_horizon
        self.n_modes = n_modes
        self.use_koopman = use_koopman
        self.learning_rate = learning_rate
        self.max_epochs = epochs
        self.device = device
        self.training_history_ = []
        self.input_shape_ = None
        self.output_shape_ = None
        self.output_dim_ = None

    def _create_dmd_data_matrix(self, trajectories):
        x_list, y_list = [], []
        for trajectory in trajectories:
            if len(trajectory) >= self.forecast_horizon + 1:
                x_input = trajectory[:-self.forecast_horizon]
                y_target = trajectory[-self.forecast_horizon:]
                x_list.append(x_input)
                y_list.append(y_target)

        x_np = np.asarray(x_list, dtype=float)
        y_np = np.asarray(y_list, dtype=float)
        self.input_shape_ = tuple(int(value) for value in x_np.shape[1:]) if x_np.ndim > 1 else ()
        self.output_shape_ = tuple(int(value) for value in y_np.shape[1:]) if y_np.ndim > 1 else ()

        x_flat = x_np.reshape(x_np.shape[0], -1)
        y_flat = y_np.reshape(y_np.shape[0], -1)
        x_torch = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
        y_torch = torch.tensor(y_flat, dtype=torch.float32, device=self.device)
        return x_torch, y_torch

    def _compute_classical_dmd(self, x_matrix, y_matrix):
        u, singular_values, vt = torch.svd(x_matrix)

        if self.n_modes is not None:
            u = u[:, :self.n_modes]
            singular_values = singular_values[:self.n_modes]
            vt = vt[:self.n_modes, :]

        singular_values_inv = torch.diag(1.0 / singular_values)
        x_pinv = vt.T @ singular_values_inv @ u.T
        operator = y_matrix @ x_pinv
        return operator, u, singular_values, vt

    def _setup_koopman_model(self, input_dim):
        if self.n_modes is None:
            self.n_modes = min(64, input_dim * 2)

        output_dim = self.output_dim_ or self.forecast_horizon
        self.K = nn.Parameter(torch.randn(self.n_modes, self.n_modes, device=self.device) * 0.01).to(self.device)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, self.n_modes),
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(self.n_modes, 256),
            nn.Tanh(),
            nn.Linear(256, output_dim),
        ).to(self.device)

    def _koopman_forward(self, x_matrix):
        z_matrix = self.encoder(x_matrix)
        z_evolved = z_matrix @ self.K.T
        return self.decoder(z_evolved)

    def fit(self, trajectory_matrix, window_size=20):
        self.window_size_ = window_size
        x_matrix, y_matrix = self._create_dmd_data_matrix(trajectory_matrix)
        self.input_dim_ = x_matrix.shape[1] if x_matrix.ndim > 1 else 1
        self.output_dim_ = y_matrix.shape[1] if y_matrix.ndim > 1 else 1
        return self._fit_koopman_dmd(x_matrix, y_matrix) if self.use_koopman else self._fit_classical_dmd(
            x_matrix, y_matrix
        )

    def _fit_classical_dmd(self, x_matrix, y_matrix):
        self.training_history_ = []
        self.A, self.U, self.S, self.Vt = self._compute_classical_dmd(x_matrix, y_matrix)
        return self

    def _fit_koopman_dmd(self, x_matrix, y_matrix):
        self._setup_koopman_model(self.input_dim_)
        self.training_history_ = []
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + [self.K]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            predictions = self._koopman_forward(x_matrix)
            loss = nn.MSELoss()(predictions, y_matrix)
            reg_loss = torch.norm(self.K, p='fro') * 0.001
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
            self.training_history_.append(float(total_loss.item()))

        return self

    def predict(self, last_trajectory):
        last_trajectory_array = np.asarray(last_trajectory, dtype=float)
        last_trajectory_tensor = torch.tensor(last_trajectory_array.reshape(1, -1), dtype=torch.float32,
                                              device=self.device)
        if self.use_koopman:
            return self._predict_koopman(last_trajectory_tensor)
        return self._predict_classical(last_trajectory_array)

    def _predict_classical(self, last_window):
        current_state = np.asarray(last_window, dtype=float).reshape(-1)
        current_torch = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        next_state = self.A @ current_torch
        prediction = next_state.detach().cpu().numpy()
        if self.output_shape_:
            return prediction.reshape(self.output_shape_)
        return prediction

    def _predict_koopman(self, last_trajectory_tensor):
        with torch.no_grad():
            predictions = self._koopman_forward(last_trajectory_tensor)
        prediction = predictions.squeeze(0).cpu().numpy()
        if self.output_shape_:
            return prediction.reshape(self.output_shape_)
        return prediction

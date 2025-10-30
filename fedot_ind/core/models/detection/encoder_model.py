from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import AnomalyDetector


class AutoencoderAnomalyDetector(AnomalyDetector):
    """
    Детектор аномалий на основе автоэнкодера
    """

    def __init__(self, contamination: float = 0.1, hidden_dim: int = 32,
                 encoding_dim: int = 8, n_epochs: int = 100, batch_size: int = 32):
        super().__init__(contamination)
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.scaler_ = StandardScaler()
        self.model_ = None
        self.reconstruction_errors_ = None

    class _Autoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, encoding_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AutoencoderAnomalyDetector':
        # Нормализация данных
        X_scaled = self.scaler_.fit_transform(X)

        # Создание модели
        input_dim = X.shape[1]
        self.model_ = self._Autoencoder(input_dim, self.hidden_dim, self.encoding_dim)

        # Обучение
        optimizer = optim.Adam(self.model_.parameters())
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch in dataloader:
                x_batch = batch[0]
                optimizer.zero_grad()
                reconstructed = self.model_(x_batch)
                loss = criterion(reconstructed, x_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Вычисление ошибок реконструкции
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            reconstructed = self.model_(X_tensor)
            self.reconstruction_errors_ = torch.mean((reconstructed - X_tensor) ** 2, axis=1).numpy()

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        X_scaled = self.scaler_.transform(X)

        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            reconstructed = self.model_(X_tensor)
            errors = torch.mean((reconstructed - X_tensor) ** 2, axis=1).numpy()

        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        threshold = np.quantile(self.reconstruction_errors_, 1 - self.contamination)
        return (scores >= threshold).astype(int)


class VAEGANAnomalyDetector(AnomalyDetector):
    """
    Гибридный детектор аномалий на основе VAE-GAN
    """

    def __init__(self, contamination: float = 0.1, latent_dim: int = 10,
                 hidden_dim: int = 64, n_epochs: int = 200):
        super().__init__(contamination)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.scaler_ = StandardScaler()

    class _Encoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2)
            )
            self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        def forward(self, x):
            h = self.fc(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

    class _Decoder(nn.Module):
        def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, z):
            return self.fc(z)

    class _Discriminator(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.fc(x)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'VAEGANAnomalyDetector':
        X_scaled = self.scaler_.fit_transform(X)
        input_dim = X.shape[1]

        # Инициализация моделей
        self.encoder_ = self._Encoder(input_dim, self.hidden_dim, self.latent_dim)
        self.decoder_ = self._Decoder(self.latent_dim, self.hidden_dim, input_dim)
        self.discriminator_ = self._Discriminator(input_dim, self.hidden_dim)

        # Оптимизаторы
        optimizer_ed = optim.Adam(
            list(self.encoder_.parameters()) + list(self.decoder_.parameters()),
            lr=1e-3
        )
        optimizer_d = optim.Adam(self.discriminator_.parameters(), lr=1e-3)

        # Функции потерь
        criterion_bce = nn.BCELoss()
        criterion_mse = nn.MSELoss()

        dataset = TensorDataset(torch.FloatTensor(X_scaled))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.n_epochs):
            for batch in dataloader:
                real_data = batch[0]
                batch_size = real_data.size(0)

                # Обучение дискриминатора
                optimizer_d.zero_grad()

                # Реальные данные
                real_labels = torch.ones(batch_size, 1)
                real_output = self.discriminator_(real_data)
                d_loss_real = criterion_bce(real_output, real_labels)

                # Сгенерированные данные
                mu, logvar = self.encoder_(real_data)
                z = self._reparameterize(mu, logvar)
                fake_data = self.decoder_(z)
                fake_labels = torch.zeros(batch_size, 1)
                fake_output = self.discriminator_(fake_data.detach())
                d_loss_fake = criterion_bce(fake_output, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()

                # Обучение энкодера-декодера
                optimizer_ed.zero_grad()

                # Reconstruction loss
                reconstruction_loss = criterion_mse(fake_data, real_data)

                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Adversarial loss
                gen_labels = torch.ones(batch_size, 1)
                gen_output = self.discriminator_(fake_data)
                g_loss = criterion_bce(gen_output, gen_labels)

                total_loss = reconstruction_loss + 0.1 * kl_loss + g_loss
                total_loss.backward()
                optimizer_ed.step()

        # Сохранение ошибок реконструкции для нормальных данных
        with torch.no_grad():
            mu, logvar = self.encoder_(torch.FloatTensor(X_scaled))
            z = self._reparameterize(mu, logvar)
            reconstructed = self.decoder_(z)
            self.reconstruction_errors_ = torch.mean((reconstructed - torch.FloatTensor(X_scaled)) ** 2, axis=1).numpy()

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        X_scaled = self.scaler_.transform(X)

        with torch.no_grad():
            mu, logvar = self.encoder_(torch.FloatTensor(X_scaled))
            z = self._reparameterize(mu, logvar)
            reconstructed = self.decoder_(z)
            errors = torch.mean((reconstructed - torch.FloatTensor(X_scaled)) ** 2, axis=1).numpy()

        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        threshold = np.quantile(self.reconstruction_errors_, 1 - self.contamination)
        return (scores >= threshold).astype(int)


class LSTMAnomalyDetector(AnomalyDetector):
    """
    Детектор аномалий на основе LSTM для прогнозирования
    """

    def __init__(self, contamination: float = 0.1, hidden_dim: int = 50,
                 n_layers: int = 2, n_epochs: int = 100, sequence_length: int = 10):
        super().__init__(contamination)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.sequence_length = sequence_length
        self.scaler_ = StandardScaler()

    class _LSTMPredictor(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, output_dim: int = 1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            output = self.fc(last_output)
            return output

    def _create_sequences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Создание последовательностей для обучения LSTM"""
        sequences = []
        targets = []

        for i in range(len(X) - self.sequence_length):
            seq = X[i:i + self.sequence_length]
            target = X[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LSTMAnomalyDetector':
        X_scaled = self.scaler_.fit_transform(X)

        # Создание последовательностей
        sequences, targets = self._create_sequences(X_scaled)

        # Инициализация модели
        self.model_ = self._LSTMPredictor(
            input_dim=X.shape[1],
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            output_dim=X.shape[1]
        )

        # Обучение
        optimizer = optim.Adam(self.model_.parameters())
        criterion = nn.MSELoss()

        dataset = TensorDataset(
            torch.FloatTensor(sequences),
            torch.FloatTensor(targets)
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_seq, batch_target in dataloader:
                optimizer.zero_grad()
                predictions = self.model_(batch_seq)
                loss = criterion(predictions, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Сохранение ошибок прогнозирования для нормальных данных
        self.model_.eval()
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequences)
            target_tensor = torch.FloatTensor(targets)
            predictions = self.model_(seq_tensor)
            self.forecast_errors_ = torch.mean((predictions - target_tensor) ** 2, axis=1).numpy()

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        X_scaled = self.scaler_.transform(X)
        sequences, targets = self._create_sequences(X_scaled)

        self.model_.eval()
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequences)
            target_tensor = torch.FloatTensor(targets)
            predictions = self.model_(seq_tensor)
            errors = torch.mean((predictions - target_tensor) ** 2, axis=1).numpy()

        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        threshold = np.quantile(self.forecast_errors_, 1 - self.contamination)
        return (scores >= threshold).astype(int)

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.covariance import EllipticEnvelope

from .base import AnomalyDetector
from .encoder_model import (LSTMAnomalyDetector)


class HybridStatisticalDLDetector(AnomalyDetector):
    """
    Гибридный детектор: статистические признаки + нейросетевая модель
    """

    def __init__(self, contamination: float = 0.1, hidden_dim: int = 64,
                 n_epochs: int = 100, use_cuda: bool = False):
        super().__init__(contamination)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.scaler_statistical = None
        self.scaler_deep = None

    class _HybridNetwork(nn.Module):
        def __init__(self, statistical_dim: int, series_dim: int, hidden_dim: int):
            super().__init__()
            # Ветка для статистических признаков
            self.statistical_branch = nn.Sequential(
                nn.Linear(statistical_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU()
            )

            # Ветка для исходных данных
            self.series_branch = nn.Sequential(
                nn.Linear(series_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )

            # Объединяющая ветка
            combined_dim = (hidden_dim // 4) + (hidden_dim // 2)
            self.combined = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        def forward(self, statistical_features: torch.Tensor, series_data: torch.Tensor) -> torch.Tensor:
            statistical_out = self.statistical_branch(statistical_features)
            series_out = self.series_branch(series_data)
            combined = torch.cat([statistical_out, series_out], dim=1)
            return self.combined(combined)

    def _extract_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """Извлечение комплексных статистических признаков"""
        features = []

        for series in X:
            series_features = []

            # Базовые статистики
            series_features.extend([
                np.mean(series), np.std(series), np.median(series),
                np.min(series), np.max(series), np.ptp(series)  # range
            ])

            # Моменты распределения
            series_features.extend([
                stats.skew(series), stats.kurtosis(series)
            ])

            # Квантили
            quantiles = np.quantile(series, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            series_features.extend(quantiles)

            # Автокорреляция
            if len(series) > 1:
                autocorr = np.correlate(series - np.mean(series),
                                        series - np.mean(series), mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
                series_features.extend(autocorr[1:4])  # лаги 1-3
            else:
                series_features.extend([0, 0, 0])

            # Энтропия
            hist, _ = np.histogram(series, bins=10, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist))
            series_features.append(entropy)

            # Стабильность (отношение std первых и последних точек)
            n_stability = min(10, len(series) // 2)
            if n_stability > 0:
                stability_ratio = np.std(series[:n_stability]) / (np.std(series[-n_stability:]) + 1e-8)
                series_features.append(stability_ratio)
            else:
                series_features.append(0)

            features.append(series_features)

        return np.array(features)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'HybridStatisticalDLDetector':
        from sklearn.preprocessing import StandardScaler

        # Извлечение и нормализация статистических признаков
        statistical_features = self._extract_statistical_features(X)
        self.scaler_statistical = StandardScaler()
        statistical_scaled = self.scaler_statistical.fit_transform(statistical_features)

        # Нормализация исходных данных
        self.scaler_deep = StandardScaler()
        series_scaled = self.scaler_deep.fit_transform(X)

        # Создание модели
        statistical_dim = statistical_scaled.shape[1]
        series_dim = series_scaled.shape[1]

        self.model_ = self._HybridNetwork(statistical_dim, series_dim, self.hidden_dim)
        self.model_.to(self.device)

        # Обучение (если есть размеченные данные)
        if y is not None:
            optimizer = optim.Adam(self.model_.parameters(), lr=1e-3)
            criterion = nn.BCELoss()

            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(statistical_scaled),
                torch.FloatTensor(series_scaled),
                torch.FloatTensor(y)
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

            self.model_.train()
            for epoch in range(self.n_epochs):
                total_loss = 0
                for stat_batch, series_batch, y_batch in dataloader:
                    stat_batch = stat_batch.to(self.device)
                    series_batch = series_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    predictions = self.model_(stat_batch, series_batch)
                    loss = criterion(predictions.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        else:
            # Если нет размеченных данных, используем статистический детектор для псевдо-разметки
            statistical_detector = EllipticEnvelope(contamination=self.contamination)
            pseudo_labels = statistical_detector.fit_predict(statistical_scaled)
            pseudo_labels = (pseudo_labels == -1).astype(int)

            # Обучение на псевдо-разметке
            optimizer = optim.Adam(self.model_.parameters(), lr=1e-3)
            criterion = nn.BCELoss()

            self.model_.train()
            for epoch in range(self.n_epochs):
                stat_tensor = torch.FloatTensor(statistical_scaled).to(self.device)
                series_tensor = torch.FloatTensor(series_scaled).to(self.device)
                labels_tensor = torch.FloatTensor(pseudo_labels).to(self.device)

                optimizer.zero_grad()
                predictions = self.model_(stat_tensor, series_tensor)
                loss = criterion(predictions.squeeze(), labels_tensor)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        statistical_features = self._extract_statistical_features(X)
        statistical_scaled = self.scaler_statistical.transform(statistical_features)
        series_scaled = self.scaler_deep.transform(X)

        self.model_.eval()
        with torch.no_grad():
            stat_tensor = torch.FloatTensor(statistical_scaled).to(self.device)
            series_tensor = torch.FloatTensor(series_scaled).to(self.device)
            scores = self.model_(stat_tensor, series_tensor)

        return scores.cpu().numpy().squeeze()

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        threshold = np.quantile(scores, 1 - self.contamination)
        return (scores >= threshold).astype(int)


class ResidualHybridDetector(AnomalyDetector):
    """
    Гибридный детектор на основе остатков LSTM + статистический анализ
    """

    def __init__(self, contamination: float = 0.1, lstm_hidden: int = 50,
                 n_epochs: int = 100):
        super().__init__(contamination)
        self.lstm_hidden = lstm_hidden
        self.n_epochs = n_epochs
        self.lstm_detector = None
        self.residual_detector = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ResidualHybridDetector':
        # LSTM для прогнозирования и получения остатков
        self.lstm_detector = LSTMAnomalyDetector(
            contamination=0.05,  # более строгий порог для остатков
            hidden_dim=self.lstm_hidden,
            n_epochs=self.n_epochs
        )

        # Обучение LSTM
        self.lstm_detector.fit(X)

        # Получение остатков (ошибок прогнозирования)
        lstm_scores = self.lstm_detector.decision_function(X)

        # Статистический анализ остатков
        self.residual_detector = EllipticEnvelope(contamination=self.contamination)

        # Преобразование остатков в признаки
        residual_features = self._extract_residual_features(lstm_scores, X)
        self.residual_detector.fit(residual_features)

        self.is_fitted = True
        return self

    def _extract_residual_features(self, residuals: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Извлечение признаков из остатков"""
        features = []

        for i, series in enumerate(X):
            series_features = []

            # Статистики остатков
            residual = residuals[i] if i < len(residuals) else 0
            series_features.extend([
                residual,
                np.abs(residual),
                residual ** 2
            ])

            # Отношение остатка к статистикам ряда
            series_mean = np.mean(series)
            series_std = np.std(series) if np.std(series) > 0 else 1e-8
            series_features.extend([
                residual / series_std,
                np.abs(residual) / series_std
            ])

            features.append(series_features)

        return np.array(features)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        # Оценка LSTM
        lstm_scores = self.lstm_detector.decision_function(X)

        # Признаки остатков
        residual_features = self._extract_residual_features(lstm_scores, X)

        # Оценка статистического детектора
        statistical_scores = -self.residual_detector.decision_function(residual_features)

        # Комбинирование оценок
        combined_scores = 0.7 * lstm_scores + 0.3 * statistical_scores

        return combined_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        threshold = np.quantile(scores, 1 - self.contamination)
        return (scores >= threshold).astype(int)

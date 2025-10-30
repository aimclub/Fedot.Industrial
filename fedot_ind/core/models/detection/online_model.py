from collections import deque
from typing import Optional, Deque

import numpy as np
import torch
import torch.nn as nn

from .base import AnomalyDetector


class OnlineAutoencoderDetector(AnomalyDetector):
    """
    Онлайн-версия автоэнкодера с адаптивным обучением
    """

    def __init__(self, contamination: float = 0.1, hidden_dim: int = 32,
                 encoding_dim: int = 8, window_size: int = 1000,
                 learning_rate: float = 0.001, adaptation_rate: float = 0.01):
        super().__init__(contamination)
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.adaptation_rate = adaptation_rate
        self.data_window: Deque[np.ndarray] = deque(maxlen=window_size)
        self.model_ = None
        self.optimizer_ = None
        self.criterion = nn.MSELoss()
        self.input_dim_ = None
        self.reconstruction_errors_ = []

    class _OnlineAutoencoder(nn.Module):
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

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OnlineAutoencoderDetector':
        """Инкрементальное обучение на новых данных"""
        if self.input_dim_ is None:
            self.input_dim_ = X.shape[1]
            self.model_ = self._OnlineAutoencoder(self.input_dim_, self.hidden_dim, self.encoding_dim)
            self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        # Добавление данных в окно
        for sample in X:
            self.data_window.append(sample)

        # Обучение на текущем окне
        if len(self.data_window) >= 10:  # минимальный размер батча
            window_data = np.array(list(self.data_window))
            self._update_model(window_data)

        self.is_fitted = True
        return self

    def _update_model(self, data: np.ndarray) -> None:
        """Обновление модели на текущем окне данных"""
        self.model_.train()

        # Небольшое количество эпох для онлайн-обучения
        for epoch in range(3):
            # Случайный батч из окна
            indices = np.random.choice(len(data), size=min(32, len(data)), replace=False)
            batch = torch.FloatTensor(data[indices])

            self.optimizer_.zero_grad()
            reconstructed = self.model_(batch)
            loss = self.criterion(reconstructed, batch)
            loss.backward()
            self.optimizer_.step()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OnlineAutoencoderDetector':
        # Для оффлайн-обучения используем partial_fit
        return self.partial_fit(X, y)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model_ is None:
            raise ValueError("Модель не обучена")

        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructed = self.model_(X_tensor)
            errors = torch.mean((reconstructed - X_tensor) ** 2, axis=1).numpy()

        # Адаптация порога на основе истории ошибок
        self.reconstruction_errors_.extend(errors)
        if len(self.reconstruction_errors_) > self.window_size:
            self.reconstruction_errors_ = self.reconstruction_errors_[-self.window_size:]

        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)

        # Динамический порог на основе скользящего окна
        if len(self.reconstruction_errors_) > 0:
            recent_errors = self.reconstruction_errors_[-min(100, len(self.reconstruction_errors_)):]
            threshold = np.quantile(recent_errors, 1 - self.contamination)
        else:
            threshold = np.quantile(scores, 1 - self.contamination)

        return (scores >= threshold).astype(int)


class AdaptiveThresholdDetector(AnomalyDetector):
    """
    Детектор с адаптивным порогом на основе статистики данных
    """

    def __init__(self, contamination: float = 0.1, window_size: int = 500,
                 drift_detection_sensitivity: float = 0.1):
        super().__init__(contamination)
        self.window_size = window_size
        self.drift_detection_sensitivity = drift_detection_sensitivity
        self.data_window: Deque[np.ndarray] = deque(maxlen=window_size)
        self.score_window: Deque[float] = deque(maxlen=window_size)
        self.current_threshold_ = None
        self.drift_detected_ = False

    def _calculate_statistical_threshold(self, scores: np.ndarray) -> float:
        """Вычисление статистического порога"""
        if len(scores) == 0:
            return 0.0

        # Используем несколько методов для надежности
        methods = [
            np.quantile(scores, 1 - self.contamination),
            np.mean(scores) + 2 * np.std(scores),
            np.median(scores) + 3 * self._mad(scores)
        ]

        return np.mean(methods)

    def _mad(self, data: np.ndarray) -> float:
        """Median Absolute Deviation"""
        median = np.median(data)
        return np.median(np.abs(data - median))

    def _detect_drift(self, new_scores: np.ndarray) -> bool:
        """Обнаружение дрейфа в данных"""
        if len(self.score_window) < 10:
            return False

        old_scores = np.array(list(self.score_window))
        new_mean = np.mean(new_scores)
        old_mean = np.mean(old_scores)

        # Простой детектор дрейфа на основе изменения среднего
        drift_magnitude = np.abs(new_mean - old_mean) / (np.std(old_scores) + 1e-8)
        return drift_magnitude > self.drift_detection_sensitivity

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                    scores: Optional[np.ndarray] = None) -> 'AdaptiveThresholdDetector':
        """Адаптация порога на новых данных"""

        if scores is None:
            # Если оценки не предоставлены, вычисляем простые статистические оценки
            scores = self._compute_basic_scores(X)

        # Обновление окон
        for score in scores:
            self.score_window.append(score)
        for sample in X:
            self.data_window.append(sample)

        # Обнаружение дрейфа
        if len(scores) > 0:
            self.drift_detected_ = self._detect_drift(scores)

            # Адаптация порога при дрейфе
            if self.drift_detected_:
                window_scores = np.array(list(self.score_window))
                self.current_threshold_ = self._calculate_statistical_threshold(window_scores)

        self.is_fitted = True
        return self

    def _compute_basic_scores(self, X: np.ndarray) -> np.ndarray:
        """Вычисление базовых оценок аномальности"""
        scores = []
        for series in X:
            # Простая оценка на основе отклонения от скользящего среднего
            if len(series) > 1:
                moving_avg = np.convolve(series, np.ones(5) / 5, mode='valid')
                if len(moving_avg) > 0:
                    last_value = series[-1]
                    avg_value = moving_avg[-1]
                    score = np.abs(last_value - avg_value) / (np.std(series) + 1e-8)
                    scores.append(score)
                else:
                    scores.append(0)
            else:
                scores.append(0)
        return np.array(scores)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AdaptiveThresholdDetector':
        scores = self._compute_basic_scores(X)
        return self.partial_fit(X, y, scores)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._compute_basic_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)

        # Используем текущий порог или вычисляем новый
        if self.current_threshold_ is None and len(self.score_window) > 0:
            window_scores = np.array(list(self.score_window))
            self.current_threshold_ = self._calculate_statistical_threshold(window_scores)

        threshold = self.current_threshold_ if self.current_threshold_ is not None else np.quantile(scores,
                                                                                                    1 - self.contamination)

        return (scores >= threshold).astype(int)

from typing import List, Optional

import numpy as np
from fedot.core.data.data import InputData
from scipy.spatial.distance import euclidean
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_random_state

from .base import AnomalyDetector

SHAPELET_DEFAULT_PARAMS = {'contamination': 0.1, 'shapelet_length': 10, 'n_shapelets': 5, 'random_state': 42}
GRADIENT_SHAPELET_DEFAULT_PARAMS = dict(contamination=0.1, shapelet_length=10, n_shapelets=3, learning_rate=0.01,
                                        n_epochs=100)


class ShapeletAnomalyDetector(AnomalyDetector):
    """
    Детектор аномалий на основе shapelet (характерных подпоследовательностей)
    """

    def __init__(self, shapelet_params: dict = SHAPELET_DEFAULT_PARAMS):
        super().__init__(shapelet_params)
        self.model_params = shapelet_params
        self.shapelets = []

    def _get_detection_features(self, input_data: InputData) -> List[np.ndarray]:
        """Извлечение shapelet из данных"""
        X = input_data.features
        rng = check_random_state(self.model_params['random_state'])
        series_length, n_series = X.shape

        for _ in range(self.model_params['n_shapelets']):
            # Случайный выбор ряда и начала shapelet
            series_idx = rng.randint(0, n_series)
            start_idx = rng.randint(0, series_length - self.model_params['shapelet_length'])

            shapelet = X[start_idx:start_idx + self.model_params['shapelet_length'], series_idx]
            self.shapelets.append(shapelet)

        return self._calculate_distances(input_data, self.shapelets)

    def _calculate_distances(self, input_data: InputData, shapelets: List[np.ndarray]) -> np.ndarray:
        """Вычисление минимальных расстояний до shapelet"""
        X = input_data.features
        series_length, n_series = X.shape
        distances = np.zeros((series_length, len(shapelets)))

        for i, series in enumerate(X.T):
            for j, shapelet in enumerate(shapelets):
                min_dist = float('inf')

                # Скользящее окно по ряду
                for k in range(len(series) - len(shapelet) + 1):
                    subsequence = series[k:k + len(shapelet)]
                    dist = euclidean(subsequence, shapelet)
                    min_dist = min(min_dist, dist)

                    distances[k, j] = min_dist

        return distances

    def build_model(self):
        self.model_impl = IsolationForest(contamination=self.contamination,
                                          random_state=self.model_params['random_state'])

    def decision_function(self, input_data: InputData) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        distances = self._calculate_distances(input_data, self.shapelets)
        scores = self.model_impl.decision_function(distances)
        return scores

    def predict(self, input_data: InputData) -> np.ndarray:
        scores = self.decision_function(input_data)
        # threshold = np.quantile(scores, self.contamination)
        return (scores < 0).astype(int)

    def predict_proba(self, input_data: InputData) -> np.ndarray:
        return self.decision_function(input_data)


class OptimizedShapeletAnomalyDetector(ShapeletAnomalyDetector):
    """
    Shapelet детектор с оптимизацией shapelet через градиентный спуск
    """

    def __init__(self, gradient_shapelet_params: dict = GRADIENT_SHAPELET_DEFAULT_PARAMS):
        super().__init__(gradient_shapelet_params)

    def _optimize_shapelet(self, X: np.ndarray) -> List[np.ndarray]:
        """Оптимизация shapelet через минимизацию расстояний"""
        n_series, series_length = X.shape
        optimized_shapelets = []

        # Инициализация случайных shapelet
        initial_shapelets = []
        for _ in range(self.n_shapelets):
            random_series = X[np.random.randint(0, n_series)]
            start = np.random.randint(0, series_length - self.shapelet_length)
            initial_shapelets.append(random_series[start:start + self.shapelet_length])

        for shapelet in initial_shapelets:
            optimized = self._gradient_descent_optimization(X, shapelet)
            optimized_shapelets.append(optimized)

        return optimized_shapelets

    def _gradient_descent_optimization(self, X: np.ndarray, initial_shapelet: np.ndarray) -> np.ndarray:
        """Градиентный спуск для оптимизации shapelet"""
        shapelet = initial_shapelet.copy()

        for epoch in range(self.n_epochs):
            total_gradient = np.zeros_like(shapelet)

            for series in X:
                # Находим наиболее близкую подпоследовательность
                min_dist = float('inf')
                best_start = 0

                for start in range(len(series) - len(shapelet) + 1):
                    subsequence = series[start:start + len(shapelet)]
                    dist = np.sum((subsequence - shapelet) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_start = start

                # Градиент по расстоянию
                closest_subsequence = series[best_start:best_start + len(shapelet)]
                gradient = 2 * (shapelet - closest_subsequence)
                total_gradient += gradient

            # Обновление shapelet
            shapelet -= self.learning_rate * total_gradient / len(X)

        return shapelet

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OptimizedShapeletAnomalyDetector':
        self.shapelets_ = self._optimize_shapelet(X)

        # Вычисление расстояний для нормальных рядов
        normal_distances = []
        for series in X:
            series_distances = []
            for shapelet in self.shapelets_:
                min_dist = float('inf')
                for start in range(len(series) - len(shapelet) + 1):
                    subsequence = series[start:start + len(shapelet)]
                    dist = euclidean(subsequence, shapelet)
                    min_dist = min(min_dist, dist)
                series_distances.append(min_dist)
            normal_distances.append(series_distances)

        self.normal_distances_ = np.array(normal_distances)
        self.threshold_ = np.quantile(
            np.mean(self.normal_distances_, axis=1),
            1 - self.contamination
        )

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        distances = []
        for series in X:
            series_dists = []
            for shapelet in self.shapelets_:
                min_dist = float('inf')
                for start in range(len(series) - len(shapelet) + 1):
                    subsequence = series[start:start + len(shapelet)]
                    dist = euclidean(subsequence, shapelet)
                    min_dist = min(min_dist, dist)
                series_dists.append(min_dist)
            distances.append(np.mean(series_dists))

        return np.array(distances)


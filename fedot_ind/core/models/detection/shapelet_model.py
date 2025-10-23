from typing import List

import numpy as np
from fedot.core.data.data import InputData
from scipy.spatial.distance import euclidean
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_random_state

from .base import AnomalyDetector
from ...repository.constanst_repository import SHAPELET_DEFAULT_PARAMS, GRADIENT_SHAPELET_DEFAULT_PARAMS


class ShapeletAnomalyDetector(AnomalyDetector):
    """
    Детектор аномалий на основе shapelet (характерных подпоследовательностей)
    """

    def __init__(self, shapelet_params: dict = SHAPELET_DEFAULT_PARAMS):
        super().__init__(shapelet_params)
        self.model_params = shapelet_params
        self.shapelets = []

    def _init_random_shapelet(self, X: np.ndarray):
        rng = check_random_state(self.model_params['random_state'])
        series_length, n_series = X.shape

        for _ in range(self.model_params['n_shapelets']):
            # Случайный выбор ряда и начала shapelet
            series_idx = rng.randint(0, n_series)
            start_idx = rng.randint(0, series_length - self.model_params['shapelet_length'])

            shapelet = X[start_idx:start_idx + self.model_params['shapelet_length'], series_idx]
            self.shapelets.append(shapelet)

    def _get_detection_features(self, X: np.ndarray) -> List[np.ndarray]:
        """Извлечение shapelet из данных"""
        self._init_random_shapelet(X)

        return self._calculate_distances(X, self.shapelets)

    def _calculate_distances(self, X: np.ndarray, shapelets: List[np.ndarray]) -> np.ndarray:
        """Вычисление минимальных расстояний до shapelet"""
        series_length, n_series = X.shape
        shapelet_list = []

        for i, series in enumerate(X.T):
            for j, shapelet in enumerate(shapelets):
                idx = list(range(len(series) - len(shapelet) + 1))
                subseq_list = [series[i:i + len(shapelet)] for i in idx]
                shapelet_list.append([euclidean(subseq, shapelet) for subseq in subseq_list])
            distances = np.array(shapelet_list).T

        return distances

    def build_model(self, input_data: InputData):
        self.model_impl = IsolationForest(contamination=self.contamination,
                                          random_state=self.model_params['random_state'])

    def decision_function(self, input_data: InputData) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        features = input_data.features if isinstance(input_data, InputData) else input_data
        distances = self._calculate_distances(features, self.shapelets)
        scores = self.model_impl.decision_function(distances)
        return scores


class OptimizedShapeletAnomalyDetector(ShapeletAnomalyDetector):
    """
    Shapelet детектор с оптимизацией shapelet через градиентный спуск
    """

    def __init__(self, gradient_shapelet_params: dict = GRADIENT_SHAPELET_DEFAULT_PARAMS):
        super().__init__(gradient_shapelet_params)

    def _optimize_shapelet(self, X: np.ndarray) -> List[np.ndarray]:
        """Оптимизация shapelet через минимизацию расстояний"""
        self._init_random_shapelet(X)
        optimized_shapelets = [self._gradient_descent_optimization(X, shapelet) for shapelet in self.shapelets]
        return optimized_shapelets

    def _gradient_descent_optimization(self, X: np.ndarray, initial_shapelet: np.ndarray) -> np.ndarray:
        """Градиентный спуск для оптимизации shapelet"""
        shapelet = initial_shapelet.copy()

        for epoch in range(self.model_params['n_epochs']):
            total_gradient = np.zeros_like(shapelet)

            for series in X.T:
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
            shapelet -= self.model_params['learning_rate'] * total_gradient / len(X)

        return shapelet

    def fit(self, input_data: InputData) -> 'OptimizedShapeletAnomalyDetector':
        X = input_data.features
        self.shapelets_ = self._optimize_shapelet(X)

        # Вычисление расстояний для нормальных рядов
        normal_distances = []
        for series in X.T:
            series_distances = []
            for shapelet in self.shapelets_:
                idx = list(range(len(series) - len(shapelet) + 1))
                subsequence = [series[start:start + len(shapelet)] for start in idx]
                series_distances.append([euclidean(subseq, shapelet) for subseq in subsequence])
            normal_distances.append(series_distances)

        self.normal_distances_ = np.array(normal_distances)
        self.threshold_ = np.quantile(np.mean(self.normal_distances_, axis=1), 1 - self.contamination)
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

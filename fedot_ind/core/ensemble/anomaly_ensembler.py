from typing import List, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fedot_ind.core.models.detection.base import AnomalyDetector
from fedot_ind.core.models.detection.encoder_model import AutoencoderAnomalyDetector, LSTMAnomalyDetector
from fedot_ind.core.models.detection.hybrid_model import HybridStatisticalDLDetector
from fedot_ind.core.models.detection.prefix_model import PrefixLengthAnomalyDetector, StateTransitionAnomalyDetector
from fedot_ind.core.models.detection.shapelet_model import ShapeletAnomalyDetector


class DynamicEnsembleDetector(AnomalyDetector):
    """
    Динамический ансамбль детекторов аномалий с автоматическим взвешиванием
    """

    def __init__(self, contamination: float = 0.1, diversity_threshold: float = 0.3):
        super().__init__(contamination)
        self.diversity_threshold = diversity_threshold
        self.detectors_ = []
        self.weights_ = None
        self.performance_metrics_ = {}

    def _create_detector_pool(self) -> List[AnomalyDetector]:
        """Создание пула разнообразных детекторов"""
        return [
            PrefixLengthAnomalyDetector(contamination=self.contamination),
            StateTransitionAnomalyDetector(contamination=self.contamination),
            ShapeletAnomalyDetector(contamination=self.contamination),
            AutoencoderAnomalyDetector(contamination=self.contamination),
            LSTMAnomalyDetector(contamination=self.contamination),
            HybridStatisticalDLDetector(contamination=self.contamination)
        ]

    def _calculate_diversity(self, predictions: np.ndarray) -> float:
        """Вычисление диверсификации предсказаний"""
        n_detectors = predictions.shape[1]
        diversity = 0

        for i in range(n_detectors):
            for j in range(i + 1, n_detectors):
                disagreement = np.mean(predictions[:, i] != predictions[:, j])
                diversity += disagreement

        return diversity / (n_detectors * (n_detectors - 1) / 2)

    def _calculate_uncertainty(self, scores: np.ndarray) -> np.ndarray:
        """Вычисление неопределенности предсказаний"""
        return np.std(scores, axis=1)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DynamicEnsembleDetector':
        # Создание и обучение детекторов
        self.detectors_ = self._create_detector_pool()

        for detector in self.detectors_:
            detector.fit(X, y)

        # Вычисление весов на основе диверсификации и производительности
        if y is not None:
            self._calculate_optimal_weights(X, y)
        else:
            # Если нет размеченных данных, используем равные веса
            n_detectors = len(self.detectors_)
            self.weights_ = np.ones(n_detectors) / n_detectors

        self.is_fitted = True
        return self

    def _calculate_optimal_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Вычисление оптимальных весов для ансамбля"""
        from sklearn.metrics import f1_score

        n_detectors = len(self.detectors_)
        performances = np.zeros(n_detectors)
        predictions = np.zeros((len(X), n_detectors))

        # Оценка производительности каждого детектора
        for i, detector in enumerate(self.detectors_):
            pred = detector.predict(X)
            predictions[:, i] = pred
            performances[i] = f1_score(y, pred)

        # Вычисление диверсификации
        diversity = self._calculate_diversity(predictions)

        # Взвешивание: производительность + диверсификация
        base_weights = performances / np.sum(performances)
        diversity_bonus = diversity * self.diversity_threshold

        # Корректировка весов с учетом диверсификации
        self.weights_ = base_weights * (1 + diversity_bonus)
        self.weights_ = self.weights_ / np.sum(self.weights_)

        # Сохранение метрик
        self.performance_metrics_ = {
            'individual_performances': performances,
            'ensemble_diversity': diversity,
            'final_weights': self.weights_.copy()
        }

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        all_scores = []
        for detector in self.detectors_:
            scores = detector.decision_function(X)
            all_scores.append(scores)

        # Взвешенное усреднение
        weighted_scores = np.zeros(len(X))
        for i, weight in enumerate(self.weights_):
            weighted_scores += weight * all_scores[i]

        return weighted_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        threshold = np.quantile(scores, 1 - self.contamination)
        return (scores >= threshold).astype(int)

    def get_detector_contributions(self) -> Dict[str, float]:
        """Получение вклада каждого детектора в ансамбль"""
        detector_names = [type(detector).__name__ for detector in self.detectors_]
        return dict(zip(detector_names, self.weights_))


class StackingAnomalyDetector(AnomalyDetector):
    """
    Стекинг ансамбль с мета-классификатором
    """

    def __init__(self, contamination: float = 0.1, meta_model: str = 'isolation_forest'):
        super().__init__(contamination)
        self.meta_model_type = meta_model
        self.base_detectors_ = []
        self.meta_detector_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StackingAnomalyDetector':
        # Базовые детекторы
        self.base_detectors_ = [
            PrefixLengthAnomalyDetector(contamination=self.contamination),
            ShapeletAnomalyDetector(contamination=self.contamination),
            AutoencoderAnomalyDetector(contamination=self.contamination),
            LSTMAnomalyDetector(contamination=self.contamination)
        ]

        # Обучение базовых детекторов и получение их предсказаний
        meta_features = []

        for detector in self.base_detectors_:
            detector.fit(X, y)
            scores = detector.decision_function(X)
            meta_features.append(scores)

        # Создание мета-признаков
        meta_features = np.column_stack(meta_features)
        meta_features_scaled = self.scaler_.fit_transform(meta_features)

        # Обучение мета-детектора
        if self.meta_model_type == 'isolation_forest':
            self.meta_detector_ = IsolationForest(contamination=self.contamination)
            self.meta_detector_.fit(meta_features_scaled)
        elif self.meta_model_type == 'elliptic_envelope':
            from sklearn.covariance import EllipticEnvelope
            self.meta_detector_ = EllipticEnvelope(contamination=self.contamination)
            self.meta_detector_.fit(meta_features_scaled)

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        # Получение предсказаний базовых детекторов
        meta_features = []
        for detector in self.base_detectors_:
            scores = detector.decision_function(X)
            meta_features.append(scores)

        meta_features = np.column_stack(meta_features)
        meta_features_scaled = self.scaler_.transform(meta_features)

        # Предсказание мета-детектора
        return -self.meta_detector_.decision_function(meta_features_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        threshold = np.quantile(scores, 1 - self.contamination)
        return (scores >= threshold).astype(int)

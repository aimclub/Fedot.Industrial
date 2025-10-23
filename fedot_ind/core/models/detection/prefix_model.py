import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .base import AnomalyDetector
from ...repository.constanst_repository import PREFIX_DEFAULT_PARAMS, STATE_TRANSITION_DEFAULT_PARAMS


class PrefixLengthAnomalyDetector(AnomalyDetector):
    """
    Детектор аномалий на основе анализа длин префиксов
    Анализирует, насколько быстро временной ряд достигает определенных состояний
    """

    def __init__(self, prefix_params: dict = PREFIX_DEFAULT_PARAMS):
        super().__init__(prefix_params)
        self.model_params = prefix_params
        self.quantile_thresholds_ = None
        self.prefix_features_ = None

    def _get_detection_features(self, X: np.ndarray) -> np.ndarray:
        """Извлечение признаков на основе префиксов"""
        features = []
        for series in X.T:
            series_features = []

            # Достижение квантилей
            quantiles = np.quantile(series, np.linspace(0.1, 0.9, self.model_params['n_quantiles']))
            for q in quantiles:
                idx = np.where(series >= q)[0]
                prefix_length = idx[0] if len(idx) > 0 else len(series)
                series_features.append(prefix_length)

            # Статистики первых n точек
            n_prefix = min(10, len(series))
            prefix_stats = [
                np.mean(series[:n_prefix]),
                np.std(series[:n_prefix]),
                stats.skew(series[:n_prefix]),
                np.median(series[:n_prefix])
            ]
            series_features.extend(prefix_stats)

            features.append(series_features)

        return np.array(features)

    def build_model(self, input_data: InputData):
        self.model_impl = IsolationForest(contamination=self.contamination,
                                          random_state=self.model_params['random_state'])


class StateTransitionAnomalyDetector(AnomalyDetector):
    """
    Детектор аномалий на основе анализа переходов между состояниями
    """

    def __init__(self, state_params: dict = STATE_TRANSITION_DEFAULT_PARAMS):
        super().__init__(state_params)
        self.model_params = state_params
        self.state_transitions_ = None
        self.state_means_ = None

    def build_model(self, input_data: InputData):
        self.model_impl = LocalOutlierFactor(n_neighbors=min(20, len(input_data.features)),
                                             contamination=self.contamination, novelty=True)

    def _get_detection_features(self, input_data: InputData) -> np.ndarray:
        """Извлечение признаков переходов между состояниями"""
        features = []
        X = input_data.features
        for series in X.T:
            states = pd.cut(series, bins=self.model_params['states'], labels=False)

            # Матрица переходов
            transition_matrix = np.zeros((self.model_params['states'], self.model_params['states']))
            for i in range(len(states) - 1):
                transition_matrix[states[i], states[i + 1]] += 1

            # Нормализация
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # избегаем деления на ноль
            transition_matrix = transition_matrix / row_sums

            # Признаки из матрицы переходов
            transition_features = transition_matrix.flatten()

            # Дополнительные статистики
            entropy = -np.sum(transition_matrix * np.log(transition_matrix + 1e-8))
            max_transition, min_transition = np.max(transition_matrix), np.min(transition_matrix)

            features.append(np.concatenate([transition_features, [entropy, max_transition, min_transition]]))

        return np.array(features).T

import numpy as np
from abc import ABC, abstractmethod


class AnomalyDetector(ABC):
    """
    Abstract base class for anomaly detection models.
    """
    @property
    @abstractmethod
    def model(self):
        return None

    @abstractmethod
    def fit(self, input_array: np.array):
        pass

    def predict(self, input_array: np.array) -> np.array:
        return self.model.predict(input_array)

from abc import ABC

import numpy as np


class PersistenceDiagramFeatureExtractor(ABC):
    def extract_feature_(self, persistence_diagram):
        pass

    def fit_transform(self, X_pd):
        return np.array([self.extract_feature_(diagram) for diagram in X_pd])

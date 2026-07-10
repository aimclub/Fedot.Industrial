from abc import ABC
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
import numpy as np
from gtda.diagrams import BettiCurve, Filtering, PersistenceEntropy, PersistenceLandscape, Scaler
from gtda.homology import VietorisRipsPersistence

from fedot_ind.core.architecture.settings.computational import backend_methods as np


class PersistenceDiagramFeatureExtractor(ABC):
    """Abstract class persistence diagrams features extractor.

    """

    def extract_feature_(self, persistence_diagram):
        pass

    def fit_transform(self, x_pd):
        return self.extract_feature_(x_pd)
    

class TopologicalFeaturesExtractor:
    def __init__(self, persistence_diagram_features: dict):
        self.persistence_diagram_features_ = persistence_diagram_features

    def transform(self, diagrams_batch: np.ndarray) -> pd.DataFrame:
        """
        Extracts features from a batch of persistence diagrams.
        Parameters:
            diagrams_batch: np.ndarray
                A batch of persistence diagrams with shape (B, K, 3), where B is the batch size, K is the number of points in each diagram, and 3 corresponds to (birth, death, dimension).
        Returns:
            pd.DataFrame
                A DataFrame containing the extracted features for each persistence diagram in the batch.
        """
        batch_size = diagrams_batch.shape[0]
        batch_features = []

        for i in range(batch_size):
            diagram = diagrams_batch[i]
            feature_list = []
            column_list = []
            
            for feature_name, feature_model in self.persistence_diagram_features_.items():
                try:
                    x_features = feature_model.fit_transform(diagram)
                    feature_list.append(x_features)
                    for dim in range(len(x_features)):
                        column_list.append(f'{feature_name}_{dim}')
                except Exception:
                    expected_len = getattr(feature_model, 'expected_len_', 3) 
                    feature_list.append(np.zeros(expected_len))
                    for dim in range(expected_len):
                        column_list.append(f'{feature_name}_{dim}')
                        
            batch_features.append(np.concatenate(feature_list))

        x_transformed = pd.DataFrame(data=np.vstack(batch_features), columns=column_list)
        return x_transformed


class HolesNumberFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(HolesNumberFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        for hole in persistence_diagram:
            if hole[1] - hole[0] > 0:
                feature[int(hole[2])] += 1.0
        return feature


class MaxHoleLifeTimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(MaxHoleLifeTimeFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        for hole in persistence_diagram:
            lifetime = hole[1] - hole[0]
            if lifetime > feature[int(hole[2])]:
                feature[int(hole[2])] = lifetime
        return feature


class RelevantHolesNumber(PersistenceDiagramFeatureExtractor):
    def __init__(self, ratio=0.7):
        super(RelevantHolesNumber).__init__()
        self.ratio_ = ratio

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        max_lifetimes = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)

        for hole in persistence_diagram:
            lifetime = hole[1] - hole[0]
            if lifetime > max_lifetimes[int(hole[2])]:
                max_lifetimes[int(hole[2])] = lifetime

        for hole in persistence_diagram:
            index = int(hole[2])
            lifetime = hole[1] - hole[0]
            if np.equal(lifetime, self.ratio_ * max_lifetimes[index]):
                feature[index] += 1.0

        return feature


class AverageHoleLifetimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(AverageHoleLifetimeFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        n_holes = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)

        for hole in persistence_diagram:
            lifetime = hole[1] - hole[0]
            index = int(hole[2])
            if lifetime > 0:
                feature[index] += lifetime
                n_holes[index] += 1

        for i in range(feature.shape[0]):
            feature[i] = feature[i] / n_holes[i] if n_holes[i] != 0 else 0.0

        return feature


class SumHoleLifetimeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(SumHoleLifetimeFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        feature = np.zeros(int(np.max(persistence_diagram[:, 2])) + 1)
        for hole in persistence_diagram:
            feature[int(hole[2])] += hole[1] - hole[0]
        return feature


class PersistenceEntropyFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(PersistenceEntropyFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        persistence_entropy = PersistenceEntropy(n_jobs=-1)
        return persistence_entropy.fit_transform([persistence_diagram])[0]


class SimultaneousAliveHolesFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(SimultaneousAliveHolesFeature).__init__()

    @staticmethod
    def get_average_intersection_number_(segments):
        intersections = list()
        n_segments = segments.shape[0]

        for i in range(n_segments):
            count = 1
            start = segments[i, 0]
            end = segments[i, 1]

            for j in range(i + 1, n_segments):
                if start <= segments[j, 0] <= end:
                    count += 1
                else:
                    break
            intersections.append(count)

        return np.sum(intersections) / len(intersections)

    def get_average_simultaneous_holes_(self, holes):
        starts = holes[:, 0]
        ends = holes[:, 1]
        ind = np.lexsort((starts, ends))
        segments = np.array([[starts[i], ends[i]] for i in ind])
        return self.get_average_intersection_number_(segments)

    def extract_feature_(self, persistence_diagram):
        n_dims = int(np.max(persistence_diagram[:, 2])) + 1
        feature = np.zeros(n_dims)

        for dim in range(n_dims):
            holes = list()
            for hole in persistence_diagram:
                if hole[1] - hole[0] != 0.0 and int(hole[2]) == dim:
                    holes.append(hole)
            if len(holes) != 0:
                feature[dim] = self.get_average_simultaneous_holes_(
                    np.array(holes))

        return feature


class AveragePersistenceLandscapeFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(AveragePersistenceLandscapeFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        # As practice shows, only 1st layer of 1st homology dimension plays
        # role
        persistence_landscape = PersistenceLandscape(
            n_jobs=-1).fit_transform([persistence_diagram])[0, 1, 0, :]
        return np.array([np.sum(persistence_landscape) /
                        persistence_landscape.shape[0]])


class BettiNumbersSumFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(BettiNumbersSumFeature).__init__()

    def extract_feature_(self, persistence_diagram):
        betti_curve = BettiCurve(
            n_jobs=-1).fit_transform([persistence_diagram])[0]
        return np.array([np.sum(betti_curve[i, :])
                        for i in range(int(np.max(persistence_diagram[:, 2])) + 1)])


class RadiusAtMaxBNFeature(PersistenceDiagramFeatureExtractor):
    def __init__(self):
        super(RadiusAtMaxBNFeature).__init__()

    def extract_feature_(self, persistence_diagram, n_bins=100):
        betti_curve = BettiCurve(
            n_jobs=-1, n_bins=n_bins).fit_transform([persistence_diagram])[0]
        max_dim = int(np.max(persistence_diagram[:, 2])) + 1
        max_bettis = np.array([np.max(betti_curve[i, :])
                               for i in range(max_dim)])
        return np.array([np.where(betti_curve[i, :] == max_bettis[i])[
            0][0] / (n_bins * max_dim) for i in range(max_dim)])

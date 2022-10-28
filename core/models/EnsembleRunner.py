import pandas as pd
import numpy as np
from core.models.ExperimentRunner import ExperimentRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.spectral.SSARunner import SSARunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalRunner


class EnsembleRunner(ExperimentRunner):
    """
    Class for performing experiments with ensemble of feature generators
        :param feature_generator_dict: dict that consists of 'generator_name': generator_class pairs
        :param list_of_generators: list of generator names that will be used for ensemble
    """

    def __init__(self, feature_generator_dict: dict = None,
                 list_of_generators=None, use_cache: bool = False):
        super().__init__(feature_generator_dict)
        self.use_cache = use_cache
        self.list_of_generators = list_of_generators
        self.generator_dict = dict(quantile=StatsRunner,
                                   window_quantile=StatsRunner,
                                   wavelet=SignalRunner,
                                   spectral=SSARunner,
                                   spectral_window=SSARunner,
                                   topological=TopologicalRunner,
                                   ensemble=EnsembleRunner)

    def get_features(self, input_data: pd.DataFrame, dataset_name: str = None, target: np.ndarray = None) -> pd.DataFrame:
        return self.ensemble_features(input_data, dataset_name)

    def ensemble_features(self, input_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """
        Extracts features using specified generator and combines them into one feature matrix
        :param input_data: pandas.DataFrame with input data
        :param dataset_name: name of dataset
        :return: matrix of features
        """
        features = list()
        for generator_name, generator in self.list_of_generators.items():
            features_df = generator.extract_features(input_data, dataset_name)
            features.append(features_df)

        return pd.concat(features, axis=1)

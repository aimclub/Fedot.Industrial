import pandas as pd

from core.models.ExperimentRunner import ExperimentRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.spectral.SSARunner import SSARunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalRunner


class EnsembleRunner(ExperimentRunner):
    """Class for performing experiments with ensemble of feature generators.

    Args:
        feature_generator_dict (dict, optional): Dictionary of feature generators consists of
        'generator_name': generator_class pairs.
        list_of_generators (list): List of feature generators.

    Attributes:
        feature_generator_dict (dict): Dictionary of feature generators.
        list_of_generators (list): List of feature generators.

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

    def get_features(self, input_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        return self.ensemble_features(input_data, dataset_name)

    def ensemble_features(self, input_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """Extracts features using specified generator and combines them into one feature matrix.

        Args:
            input_data (pd.DataFrame): Dataframe with time series data.
            dataset_name (str): Dataset name.

        Returns:
            pd.DataFrame: Dataframe with extracted features.

        """
        features = list()
        for generator_name, generator in self.list_of_generators.items():
            features_df = generator.extract_features(input_data, dataset_name)
            features.append(features_df)

        return pd.concat(features, axis=1)

import pandas as pd
import numpy as np
from core.models.ExperimentRunner import ExperimentRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.spectral.SSARunner import SSARunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalExtractor


class EnsembleRunner(ExperimentRunner):
    """Class for performing experiments with ensemble of feature generators.

    Args:
        feature_generator_dict: Dictionary of feature generators consists of
            'generator_name': generator_class pairs.
        list_of_generators: List of feature generators.

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
                                   topological=TopologicalExtractor,
                                   ensemble=EnsembleRunner)

    def get_features(self, input_data: pd.DataFrame, dataset_name: str = None, target: np.ndarray = None) -> pd.DataFrame:
        return self.ensemble_features(input_data, dataset_name)

    def ensemble_features(self, input_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """Extracts features using specified generator and combines them into one feature matrix.

        Args:
            input_data: Dataframe with time series data.
            dataset_name: Dataset name.

        Returns:
            Dataframe with extracted features.

        """
        features = list()
        for generator_name, generator in self.list_of_generators.items():
            features_df = generator.extract_features(input_data, dataset_name)
            features.append(features_df)

        return pd.concat(features, axis=1)

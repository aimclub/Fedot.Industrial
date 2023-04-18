import pandas as pd

from fedot_ind.core.models.BaseExtractor import BaseExtractor
from fedot_ind.core.models.signal.SignalExtractor import SignalExtractor
from fedot_ind.core.models.spectral.SSAExtractor import SSAExtractor
from fedot_ind.core.models.statistical.StatsExtractor import StatsExtractor
from fedot_ind.core.models.topological.TopologicalExtractor import TopologicalExtractor


class EnsembleExtractor(BaseExtractor):
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
        self.generator_dict = dict(quantile=StatsExtractor,
                                   window_quantile=StatsExtractor,
                                   wavelet=SignalExtractor,
                                   spectral=SSAExtractor,
                                   spectral_window=SSAExtractor,
                                   topological=TopologicalExtractor,
                                   ensemble=EnsembleExtractor)

    # def get_features(self, ts_frame: pd.DataFrame, dataset_name: str = None, target: np.ndarray = None) -> pd.DataFrame:
    #     return self.ensemble_features(ts_frame, dataset_name)

    def generate_features_from_ts(self, input_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """Extracts features using specified generator and combines them into one feature matrix.

        Args:
            input_data: Dataframe with time series data.
            dataset_name: Dataset name.

        Returns:
            Dataframe with extracted features.

        """
        self.logger.info(f'Extracting features using ensemble of generators: {self.list_of_generators.keys()}')
        features = list()
        for generator_name, generator in self.list_of_generators.items():
            features_df = generator.extract_features(input_data, dataset_name)
            features.append(features_df)

        self.logger.info(f'Features extraction finished')
        return pd.concat(features, axis=1)

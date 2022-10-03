import timeit

from gtda.time_series import SingleTakensEmbedding

from core.models.ExperimentRunner import ExperimentRunner
from core.models.topological.TDA import Topological
from core.models.topological.TFE import *

dict_of_dataset = dict
dict_of_win_list = dict

PERSISTENCE_DIAGRAM_FEATURES = {'HolesNumberFeature': HolesNumberFeature(),
                                'MaxHoleLifeTimeFeature': MaxHoleLifeTimeFeature(),
                                'RelevantHolesNumber': RelevantHolesNumber(),
                                'AverageHoleLifetimeFeature': AverageHoleLifetimeFeature(),
                                'SumHoleLifetimeFeature': SumHoleLifetimeFeature(),
                                'PersistenceEntropyFeature': PersistenceEntropyFeature(),
                                'SimultaneousAliveHolesFeature': SimultaneousAliveHolesFeature(),
                                'AveragePersistenceLandscapeFeature': AveragePersistenceLandscapeFeature(),
                                'BettiNumbersSumFeature': BettiNumbersSumFeature(),
                                'RadiusAtMaxBNFeature': RadiusAtMaxBNFeature()}


class TopologicalRunner(ExperimentRunner):
    """
    Class for extracting topological features from time series data
        :param list_of_dataset: list of dataset names that will be used for experiments
    """
    def __init__(self, list_of_dataset: list = None,
                 use_cache: bool = False):
        super().__init__(list_of_dataset)
        self.use_cache = use_cache
        self.TE_dimension = None
        self.TE_time_delay = None

    def generate_topological_features(self, ts_data: pd.DataFrame) -> pd.DataFrame:

        if not self.TE_dimension and not self.TE_time_delay:
            single_ts = ts_data.loc[0]
            self.TE_dimension, self.TE_time_delay = self.get_embedding_params(single_time_series=single_ts)
            self.logger.info(f'TE_delay: {self.TE_time_delay}, TE_dimension: {self.TE_dimension} are selected')

        persistence_diagram_extractor = PersistenceDiagramsExtractor(takens_embedding_dim=self.TE_dimension,
                                                                     takens_embedding_delay=self.TE_time_delay,
                                                                     homology_dimensions=(0, 1),
                                                                     parallel=True)

        feature_extractor = TopologicalFeaturesExtractor(persistence_diagram_extractor=persistence_diagram_extractor,
                                                         persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

        ts_data_transformed = feature_extractor.fit_transform(ts_data.values)
        ts_data_transformed = self.delete_col_by_var(ts_data_transformed)

        return ts_data_transformed

    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None):
        return self.generate_topological_features(ts_data=ts_data)

    @staticmethod
    def get_embedding_params(single_time_series):
        """
        Method for getting optimal Takens embedding parameters.

        :param single_time_series: single time series from dataset
        :return: optimal dimension and time delay
        """
        embedder = SingleTakensEmbedding(parameters_type="search",
                                         time_delay=10,
                                         dimension=10)
        embedder.fit_transform(single_time_series)
        return embedder.dimension_, embedder.time_delay_

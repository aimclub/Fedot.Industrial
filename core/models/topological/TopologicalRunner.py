from gtda.time_series import takens_embedding_optimal_parameters
from scipy import stats
from tqdm import tqdm

from core.models.ExperimentRunner import ExperimentRunner
from core.models.topological.TFE import *
from core.operation.utils.Decorators import time_it

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
    """Class for extracting topological features from time series data.

    Args:
        use_cache: flag for using cache

    """

    def __init__(self, use_cache: bool = False):
        super().__init__()
        self.use_cache = use_cache

    def generate_topological_features(self, ts_data: pd.DataFrame) -> pd.DataFrame:
        te_dimension, te_time_delay = self.get_embedding_params_from_batch(ts_data=ts_data)

        persistence_diagram_extractor = PersistenceDiagramsExtractor(takens_embedding_dim=te_dimension,
                                                                     takens_embedding_delay=te_time_delay,
                                                                     homology_dimensions=(0, 1),
                                                                     parallel=True)

        feature_extractor = TopologicalFeaturesExtractor(persistence_diagram_extractor=persistence_diagram_extractor,
                                                         persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

        ts_data_transformed = feature_extractor.fit_transform(ts_data.values)
        ts_data_transformed = self.delete_col_by_var(ts_data_transformed)

        return ts_data_transformed

    @time_it
    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None, target: np.ndarray = None):
        return self.generate_topological_features(ts_data=ts_data)

    def get_embedding_params_from_batch(self, ts_data: pd.DataFrame, method: str = 'mean') -> tuple:
        """Method for getting optimal Takens embedding parameters.

        Args:
            ts_data: dataframe with time series data
            method: method for getting optimal parameters

        Returns:
            Optimal Takens embedding parameters

        """
        methods = {'mode': self._mode,
                   'mean': np.mean,
                   'median': np.median}

        self.logger.info('Start searching optimal TE parameters')
        dim_list, delay_list = list(), list()

        for _ in tqdm(range(len(ts_data)),
                      initial=0,
                      desc='Time series processed: ',
                      unit='ts', colour='black'):
            single_time_series = ts_data.sample(1, replace=False, axis=0).squeeze()
            delay, dim = takens_embedding_optimal_parameters(X=single_time_series.values,
                                                             max_time_delay=10,
                                                             max_dimension=10,
                                                             n_jobs=-1)
            delay_list.append(delay)
            dim_list.append(dim)

        _dimension = int(methods[method](dim_list))
        _delay = int(methods[method](delay_list))
        self.logger.info(f'Optimal TE parameters: dimension = {_dimension}, time_delay = {_delay}')

        return _dimension, _delay

    @staticmethod
    def _mode(arr: list) -> int:
        return int(stats.mode(arr)[0][0])

import gc
import sys
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from gtda.time_series import takens_embedding_optimal_parameters
from scipy import stats
from tqdm import tqdm

from core.models.BaseExtractor import BaseExtractor
from core.operation.transformation.extraction.topological import *

sys.setrecursionlimit(1000000000)

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


class TopologicalExtractor(BaseExtractor):
    """Class for extracting topological features from time series data.

    Args:
        params: parameters for topological features extraction – ``use_cache``, ``max_te_dimension``,
                ``max_te_time_delay``, ``stride``.

    Notes:
        Params for topological features extraction are used to define the optimal embedding parameters for the
        Takens embedding algorithm. More you can read here:
        https://giotto-ai.github.io/gtda-docs/0.5.1/modules/generated/time_series/embedding/gtda.time_series.takens_embedding_optimal_parameters.html

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.filtered_features = None
        self.feature_extractor = None
        self.max_te_dimension = params.get('te_dimension')
        self.max_te_time_delay = params.get('te_time_delay')
        self.stride = params.get('stride')
        self.te_dimension = None
        self.te_time_delay = None

    def fit(self, input_data: InputData) -> OutputData:
        pass

    def generate_topological_features(self, ts_data: pd.DataFrame) -> pd.DataFrame:

        if not all([self.te_time_delay, self.te_dimension]):
            self.te_dimension, self.te_time_delay = self.get_embedding_params_from_batch(ts_data=ts_data)

        persistence_diagram_extractor = PersistenceDiagramsExtractor(takens_embedding_dim=self.te_dimension,
                                                                     takens_embedding_delay=self.te_time_delay,
                                                                     homology_dimensions=(0, 1),
                                                                     parallel=True)

        self.feature_extractor = TopologicalFeaturesExtractor(
            persistence_diagram_extractor=persistence_diagram_extractor,
            persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

        ts_data_transformed = self.feature_extractor.fit_transform(ts_data)
        gc.collect()
        return ts_data_transformed

    def generate_features_from_ts(self, ts_data: pd.DataFrame, dataset_name: str = None):
        return self.generate_topological_features(ts_data=ts_data)

    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None):
        self.logger.info('Topological features extraction started')
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

        self.logger.info('Searching optimal Takens embedding parameters')
        dim_list, delay_list = list(), list()

        for _ in tqdm(range(len(ts_data)),
                      initial=0,
                      desc='Time series processed: ',
                      unit='ts', colour='black'):
            ts_data = pd.DataFrame(ts_data)
            single_time_series = ts_data.sample(1, replace=False, axis=0).squeeze()
            delay, dim = takens_embedding_optimal_parameters(X=single_time_series,
                                                             max_time_delay=self.max_te_time_delay,
                                                             max_dimension=self.max_te_dimension,
                                                             stride=self.stride,
                                                             n_jobs=-1)
            delay_list.append(delay)
            dim_list.append(dim)

        dimension = int(methods[method](dim_list))
        delay = int(methods[method](delay_list))
        self.logger.info(f'Optimal TE parameters: dimension = {dimension}, time_delay = {delay}')

        return dimension, delay

    @staticmethod
    def _mode(arr: list) -> int:
        return int(stats.mode(arr)[0][0])

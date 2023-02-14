import gc
import sys
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import default_log
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from gtda.time_series import takens_embedding_optimal_parameters
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from core.architecture.abstraction.Decorators import time_it
from core.metrics.metrics_implementation import ROCAUC
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


# TODO after full refactoring inherit it from Experiment runner and ExperimentRunner from dataOperationImplementation
class TopologicalExtractor(DataOperationImplementation):
    """Class for extracting topological features from time series data.

    Args:
        use_cache: flag for using cache

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.logger = default_log()
        self.use_cache = params.get('use_cache')
        self.filtered_features = None
        self.feature_extractor = None

    def fit(self, input_data: InputData) -> OutputData:
        pass

    def transform(self, input_data: InputData):
        res = []
        for series in input_data.features:
            res.append(self.generate_topological_features(series))
        res = np.array(res)
        res = res.reshape(res.shape[0], -1)
        return res

    def generate_topological_features(self, ts_data: pd.DataFrame) -> pd.DataFrame:

        if self.feature_extractor is None:
            ts_data = pd.DataFrame(ts_data)
            te_dimension, te_time_delay = self.get_embedding_params_from_batch(ts_data=ts_data)

            persistence_diagram_extractor = PersistenceDiagramsExtractor(takens_embedding_dim=te_dimension,
                                                                         takens_embedding_delay=te_time_delay,
                                                                         homology_dimensions=(0, 1),
                                                                         parallel=True)

            self.feature_extractor = TopologicalFeaturesExtractor(
                persistence_diagram_extractor=persistence_diagram_extractor,
                persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

        ts_data_transformed = self.feature_extractor.fit_transform(ts_data)

        if self.filtered_features is None:
            ts_data_transformed = self.delete_col_by_var(ts_data_transformed)
            self.filtered_features = ts_data_transformed.columns.tolist()
        gc.collect()
        return ts_data_transformed[self.filtered_features]

    @time_it
    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None):
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
            delay, dim = takens_embedding_optimal_parameters(X=single_time_series,
                                                             max_time_delay=5,
                                                             max_dimension=5,
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

    def get_roc_auc_score(self, prediction_labels, test_labels):
        metric_roc = ROCAUC(target=test_labels, predicted_labels=prediction_labels)
        try:
            score_roc_auc = metric_roc.metric()
        except ValueError:
            self.logger.info(f'ValueError in roc_auc_score')
            score_roc_auc = 0
        return score_roc_auc

    @staticmethod
    def delete_col_by_var(dataframe: pd.DataFrame):
        for col in dataframe.columns:
            scaled_feature = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe[col].values.reshape(-1, 1))[:,
                             0]
            deviation = np.std(scaled_feature)
            if deviation < 0.05 and not col.startswith('diff'):
                del dataframe[col]
        return dataframe

    @staticmethod
    def apply_window_for_stat_feature(ts_data: pd.DataFrame,
                                      feature_generator: callable,
                                      window_size: int = None):
        ts_data = ts_data.T
        if window_size is None:
            window_size = round(ts_data.shape[1] / 10)
        else:
            window_size = round(ts_data.shape[1] / window_size)
        tmp_list = []
        for i in range(0, ts_data.shape[1], window_size):
            slice_ts = ts_data.iloc[:, i:i + window_size]
            if slice_ts.shape[1] == 1:
                break
            else:
                df = feature_generator(slice_ts)
                df.columns = [x + f'_on_interval: {i} - {i + window_size}' for x in df.columns]
                tmp_list.append(df)
        return tmp_list

import gc
import sys
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from gtda.time_series import takens_embedding_optimal_parameters
from scipy import stats
from tqdm import tqdm

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.models.topological.topofeatures import AverageHoleLifetimeFeature, \
    AveragePersistenceLandscapeFeature, BettiNumbersSumFeature, HolesNumberFeature, MaxHoleLifeTimeFeature, \
    PersistenceDiagramsExtractor, PersistenceEntropyFeature, RadiusAtMaxBNFeature, RelevantHolesNumber, \
    SimultaneousAliveHolesFeature, SumHoleLifetimeFeature, TopologicalFeaturesExtractor
from fedot_ind.core.operation.transformation.data.point_cloud import TopologicalTransformation

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

PERSISTENCE_DIAGRAM_EXTRACTOR = PersistenceDiagramsExtractor(takens_embedding_dim=1,
                                                             takens_embedding_delay=2,
                                                             homology_dimensions=(0, 1),
                                                             parallel=False)


class TopologicalExtractor(BaseExtractor):
    """Class for extracting topological features from time series data.

    Args:
        params: parameters for operation

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from fedot_ind.api.utils.input_data import init_input_data
            from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('eigen_basis').add_node('topological_extractor').add_node(
                    'rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 10)
        self.stride = params.get('window_size', 1)
        self.feature_extractor = TopologicalFeaturesExtractor(
            persistence_diagram_extractor=PERSISTENCE_DIAGRAM_EXTRACTOR,
            persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)
        self.data_transformer = None

    def __evaluate_persistence_params(self, ts_data: np.array) -> InputData:
        if self.feature_extractor is None:
            te_dimension, te_time_delay = self.get_embedding_params_from_batch(ts_data=ts_data)

            persistence_diagram_extractor = PersistenceDiagramsExtractor(takens_embedding_dim=te_dimension,
                                                                         takens_embedding_delay=te_time_delay,
                                                                         homology_dimensions=(0, 1, 2),
                                                                         parallel=True)

            self.feature_extractor = TopologicalFeaturesExtractor(
                persistence_diagram_extractor=persistence_diagram_extractor,
                persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

    def _generate_features_from_ts(self, ts_data: np.array,
                                   persistence_params: dict) -> InputData:
        if self.data_transformer is None:
            self.data_transformer = TopologicalTransformation(
                persistence_params=persistence_params,
                window_length=round(ts_data.shape[0] * 0.01 * self.window_size))

        point_cloud = self.data_transformer.time_series_to_point_cloud(input_data=ts_data)
        topological_features = self.feature_extractor.transform(point_cloud)
        topological_features = InputData(idx=np.arange(len(topological_features.values)),
                                         features=topological_features.values,
                                         target='no_target',
                                         task='no_task',
                                         data_type=DataTypesEnum.table,
                                         supplementary_data={'feature_name': topological_features.columns})
        return topological_features

    def generate_topological_features(self, ts: np.array,
                                      persistence_params: dict = None) -> InputData:

        if persistence_params is not None:
            self.__evaluate_persistence_params(ts)

        if len(ts.shape) == 1:
            aggregation_df = self._generate_features_from_ts(ts, persistence_params)
        else:
            aggregation_df = self._get_feature_matrix(partial(self._generate_features_from_ts,
                                                              persistence_params=persistence_params), ts)

        return aggregation_df

    def generate_features_from_ts(self, ts_data: np.array, dataset_name: str = None):
        return self.generate_topological_features(ts=ts_data)

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

        dim_list, delay_list = list(), list()

        for _ in tqdm(range(len(ts_data)),
                      initial=0,
                      desc='Time series processed: ',
                      unit='ts', colour='black'):
            ts_data = pd.DataFrame(ts_data)
            single_time_series = ts_data.sample(1, replace=False, axis=0).squeeze()
            delay, dim = takens_embedding_optimal_parameters(X=single_time_series,
                                                             max_time_delay=1,
                                                             max_dimension=5,
                                                             n_jobs=-1)
            delay_list.append(delay)
            dim_list.append(dim)

        dimension = int(methods[method](dim_list))
        delay = int(methods[method](delay_list))

        return dimension, delay

    @staticmethod
    def _mode(arr: list) -> int:
        return int(stats.mode(arr)[0][0])

import gc
import sys
from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from gtda.time_series import takens_embedding_optimal_parameters
from scipy import stats
from tqdm import tqdm

from examples.fedot.fedot_ex import init_input_data
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.models.BaseExtractor import BaseExtractor
from fedot_ind.core.models.topological.topofeatures import AverageHoleLifetimeFeature, \
    AveragePersistenceLandscapeFeature, BettiNumbersSumFeature, HolesNumberFeature, MaxHoleLifeTimeFeature, \
    PersistenceDiagramsExtractor, PersistenceEntropyFeature, RadiusAtMaxBNFeature, RelevantHolesNumber, \
    SimultaneousAliveHolesFeature, SumHoleLifetimeFeature, TopologicalFeaturesExtractor
from fedot_ind.core.operation.transformation.data.point_cloud import TopologicalTransformation
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

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

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 10)
        self.feature_extractor = TopologicalFeaturesExtractor(
            persistence_diagram_extractor=PERSISTENCE_DIAGRAM_EXTRACTOR,
            persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)
        self.data_transformer = None

    def __evaluate_persistence_params(self, ts_data):
        if self.feature_extractor is None:
            te_dimension, te_time_delay = self.get_embedding_params_from_batch(ts_data=ts_data)

            persistence_diagram_extractor = PersistenceDiagramsExtractor(takens_embedding_dim=te_dimension,
                                                                         takens_embedding_delay=te_time_delay,
                                                                         homology_dimensions=(0, 1, 2),
                                                                         parallel=True)

            self.feature_extractor = TopologicalFeaturesExtractor(
                persistence_diagram_extractor=persistence_diagram_extractor,
                persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

    def fit(self, input_data: InputData) -> OutputData:
        pass

    def _generate_features_from_ts(self, ts_data, persistence_params):
        if self.data_transformer is None:
            self.data_transformer = TopologicalTransformation(
                persistence_params=persistence_params,
                window_length=round(ts_data.shape[0] * 0.01 * self.window_size))

        point_cloud = self.data_transformer.time_series_to_point_cloud(input_data=ts_data)
        topological_features = self.feature_extractor.transform(point_cloud)
        return topological_features

    def generate_topological_features(self, ts_data: np.array,
                                      persistence_params: dict = None) -> pd.DataFrame:

        if persistence_params is not None:
            self._evaluate_persistence_params(ts_data)

        if len(ts_data.shape) > 1:
            topological_features = [self._generate_features_from_ts(component, persistence_params)
                                    for component in ts_data]
            for component_idx, feature_df in enumerate(topological_features):
                feature_df.columns = [f'{col}_component_{component_idx}' for col in feature_df.columns]
            return pd.concat(topological_features, axis=1)

        else:
            return self._generate_features_from_ts(ts_data, persistence_params)

    def generate_features_from_ts(self, ts_data: pd.DataFrame, dataset_name: str = None):
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


# if __name__ == "__main__":
#     train_data, test_data = DataLoader(dataset_name='Ham').load_data()
#     with IndustrialModels():
#         pipeline = PipelineBuilder().add_node('data_driven_basis').add_node('topological_extractor').add_node(
#             'rf').build()
#         input_data = init_input_data(train_data[0], train_data[1])
#         pipeline.fit(input_data)
#         features = pipeline.predict(input_data)
#         print(features)

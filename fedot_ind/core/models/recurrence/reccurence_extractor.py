from functools import partial
from multiprocessing import Pool
from typing import Optional

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from sklearn.preprocessing import StandardScaler
from fedot.core.operations.operation_parameters import OperationParameters
from joblib import Parallel, delayed
from tqdm import tqdm
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.data.kernel_matrix import TSTransformer
from fedot_ind.core.models.recurrence.sequences import ReccurenceFeaturesExtractor


class RecurrenceExtractor(BaseExtractor):
    """Class responsible for wavelet feature generator experiment.
    Args:
        window_mode: boolean flag - if True, window mode is used. Defaults to False.
        use_cache: boolean flag - if True, cache is used. Defaults to False.
    Attributes:
        transformer: TSTransformer object.
        self.extractor: ReccurenceExtractor object.
        train_feats: train features.
        test_feats: test features.
    Example:
        from fedot.core.pipelines.pipeline_builder import PipelineBuilder
        from examples.fedot.fedot_ex import init_input_data
        from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
        from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

        train_data, test_data = DataLoader(dataset_name='Ham').load_data()
        with IndustrialModels():
            pipeline = PipelineBuilder().add_node('eigen_basis').add_node('recurrence_extractor').add_node(
                'rf').build()
            input_data = init_input_data(train_data[0], train_data[1])
            pipeline.fit(input_data)
            features = pipeline.predict(input_data)
            print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.image_mode = False

        self.window_mode = params.get('window_mode')
        self.min_signal_ratio = params.get('min_signal_ratio')
        self.max_signal_ratio = params.get('max_signal_ratio')
        self.rec_metric = params.get('rec_metric')
        self.window_size = 10
        self.transformer = TSTransformer
        self.extractor = ReccurenceFeaturesExtractor

    def _generate_features_from_ts(self, ts: np.array):

        if self.window_mode:
            trajectory_transformer = HankelMatrix(time_series=ts, window_size=self.window_size)
            ts = trajectory_transformer.trajectory_matrix
            self.ts_length = trajectory_transformer.ts_length

        specter = self.transformer(time_series=ts,
                                   min_signal_ratio=self.min_signal_ratio,
                                   max_signal_ratio=self.max_signal_ratio,
                                   rec_metric=self.rec_metric)
        feature_df = specter.ts_to_recurrence_matrix()

        if not self.image_mode:
            feature_df = self.extractor(recurrence_matrix=feature_df).recurrence_quantification_analysis()

        features = np.nan_to_num(np.array(list(feature_df.values())))
        recurrence_features = InputData(idx=np.arange(len(features)),
                                        features=features,
                                        target='no_target',
                                        task='no_task',
                                        data_type=DataTypesEnum.table,
                                        supplementary_data={'feature_name': list(feature_df.keys())})
        return recurrence_features

    def generate_reccurence_features(self, ts: np.array) -> InputData:

        if len(ts.shape) == 1:
            aggregation_df = self._generate_features_from_ts(ts)
        else:
            aggregation_df = self._get_feature_matrix(self._generate_features_from_ts, ts)

        return aggregation_df

    def generate_features_from_ts(self, ts_data: np.array,
                                  dataset_name: str = None):
        return self.generate_reccurence_features(ts=ts_data)
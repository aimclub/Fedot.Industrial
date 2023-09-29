from itertools import chain
from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from pandas import Index

from fedot_ind.core.models.base_extractor import BaseExtractor


class QuantileExtractor(BaseExtractor):
    """Class responsible for quantile feature generator experiment.

    Attributes:
        window_mode (bool): flag to use window or not
        window_size (int): size of window
        var_threshold (float): threshold for variance

    Example:
        To use this class you need to import it and call needed methods::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('quantile_extractor',
                                                      params={'window_size': 20,
                                                              'window_mode': True}).add_node(
                    'rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 0)
        self.stride = params.get('stride', 1)
        self.var_threshold = 0.1
        self.logging_params.update({'Wsize': self.window_size,
                                    'Stride': self.stride,
                                    'VarTh': self.var_threshold})

    def drop_features(self, predict: pd.DataFrame, columns: Index, n_components: int):
        """
        Method for dropping features with low variance
        """
        # Fill columns names for every extracted ts component
        predict = pd.DataFrame(predict,
                               columns=[f'{col}{str(i)}' for i in range(1, n_components + 1) for col in columns])

        if self.relevant_features is None:
            reduced_df, self.relevant_features = self.filter_by_var(predict, threshold=self.var_threshold)
            return reduced_df
        else:
            return predict[self.relevant_features]

    def filter_by_var(self, data: pd.DataFrame, threshold: float):
        cols = data.columns
        filtrat = {}

        for col in cols:
            if np.var(data[col].values) > threshold:
                filtrat.update({col: data[col].values.flatten()})

        return pd.DataFrame(filtrat), list(filtrat.keys())

    def _concatenate_global_and_local_feature(self, global_features: InputData,
                                              window_stat_features: InputData) -> InputData:
        if type(window_stat_features.features) is list:
            window_stat_features.features = np.concatenate(window_stat_features.features, axis=0)
            window_stat_features.supplementary_data['feature_name'] = list(
                chain(*window_stat_features.supplementary_data['feature_name']))

        window_stat_features.features = np.concatenate([global_features.features,
                                                        window_stat_features.features],
                                                       axis=0)
        window_stat_features.features = np.nan_to_num(window_stat_features.features)

        window_stat_features.supplementary_data['feature_name'] = list(
            chain(*[global_features.supplementary_data['feature_name'],
                    window_stat_features.supplementary_data['feature_name']]))
        return window_stat_features

    def extract_stats_features(self, ts: np.array) -> InputData:
        global_features = self.get_statistical_features(ts, add_global_features=True)
        if self.window_size != 0:
            window_stat_features = self.apply_window_for_stat_feature(ts_data=ts,
                                                                      feature_generator=self.get_statistical_features,
                                                                      window_size=self.window_size)
        else:
            window_stat_features = self.get_statistical_features(ts)

        return self._concatenate_global_and_local_feature(global_features, window_stat_features)

    def generate_features_from_ts(self,
                                  ts: np.array,
                                  window_length: int = None) -> InputData:

        ts = np.nan_to_num(ts)
        if len(ts.shape) == 2 and ts.shape[1] == 1:
            ts = ts.flatten()

        if len(ts.shape) == 1:
            aggregation_df = self.extract_stats_features(ts)
        else:
            aggregation_df = self._get_feature_matrix(self.extract_stats_features, ts)

        return aggregation_df

from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.BaseExtractor import BaseExtractor
from fedot_ind.core.operation.transformation.extraction.statistical import StatFeaturesExtractor


class StatsExtractor(BaseExtractor):
    """Class responsible for quantile feature generator experiment.

    Attributes:
        window_mode: Flag for window mode. Defaults to False.
        use_cache: Flag for cache usage. Defaults to False.
        use_cache (bool): Flag for cache usage.
        aggregator (StatFeaturesExtractor): StatFeaturesExtractor object.
        vis_flag (bool): Flag for visualization.
        train_feats (pd.DataFrame): Train features.
        test_feats (pd.DataFrame): Test features.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.aggregator = StatFeaturesExtractor()
        self.window_mode = params.get('window_mode')
        self.window_size = params.get('window_size')
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None

        self.logging_params.update({'Wsize': self.window_size, 'Wmode': self.window_mode})

    def fit(self, input_data: InputData):
        pass

    def extract_stats_features(self, ts):
        if self.window_mode:
            aggregator = self.aggregator.create_baseline_features
            list_of_stat_features_on_interval = self.apply_window_for_stat_feature(ts_data=ts,
                                                                                   feature_generator=aggregator,
                                                                                   window_size=self.window_size)
            aggregation_df = pd.concat(list_of_stat_features_on_interval, axis=1)
        else:
            aggregation_df = self.aggregator.create_baseline_features(ts)
        return aggregation_df

    def generate_features_from_ts(self, ts_frame: np.array) -> pd.DataFrame:

        # if ts_frame.shape[0] > 1:
        #     ts = pd.DataFrame(ts_frame, dtype=float)
        # else:
        #     ts = pd.DataFrame(ts_frame, dtype=float).T

        # if 1D vector
        if len(ts_frame.shape) == 1:
            ts = pd.DataFrame(ts_frame, dtype=float).T

        # if n-D matrix
        else:
            ts = pd.DataFrame(ts_frame, dtype=float)

        ts = ts.fillna(method='ffill')

        if ts.shape[0] == 1 or ts.shape[1] == 1:
            mode = 'SingleTS'
        else:
            mode = 'MultiTS'
        try:
            if mode == 'MultiTS':
                aggregation_df = self.__get_feature_matrix(ts)
            else:
                aggregation_df = self.extract_stats_features(ts)
        except Exception:
            aggregation_df = self.__component_extraction(ts)

        return aggregation_df

    def __component_extraction(self, ts):
        ts_components = [pd.DataFrame(x) for x in ts.T.values.tolist()]
        tmp_list = []
        for index, component in enumerate(ts_components):
            aggregation_df = self.extract_stats_features(component)
            col_name = [f'{x} for component {index}' for x in aggregation_df.columns]
            aggregation_df.columns = col_name
            tmp_list.append(aggregation_df)
        aggregation_df = pd.concat(tmp_list, axis=1)
        return aggregation_df

    def __get_feature_matrix(self, ts):
        ts_components = [pd.DataFrame(x) for x in ts.values.tolist()]
        if ts_components[0].shape[0] != 1:
            ts_components = [x.T for x in ts_components]

        tmp_list = [self.extract_stats_features(x) for x in ts_components]
        aggregation_df = pd.concat(tmp_list, axis=1)
        return aggregation_df

    def apply_window_for_stat_feature(self, ts_data: pd.DataFrame,
                                      feature_generator: callable,
                                      window_size: int = None):
        if window_size is None:
            # 10% of time series length by default
            # window_size = round(ts_data.shape[0] / 10)
            window_size = round(ts_data.shape[1] / 10)
        else:
            window_size = round(ts_data.shape[1] * (window_size / 100))
            # window_size = round(ts_data.shape[0] * (window_size / 100))
        tmp_list = []
        # for i in range(0, ts_data.shape[0], window_size):
        for i in range(0, ts_data.shape[1], window_size):
            slice_ts = ts_data.iloc[:, i:i + window_size]
            # slice_ts = ts_data.iloc[i:i + window_size, :]
            if slice_ts.shape[1] == 1:
                break
            else:
                df = feature_generator(slice_ts)
                df.columns = [x + f'_on_interval: {i} - {i + window_size}' for x in df.columns]
                tmp_list.append(df)
        return tmp_list

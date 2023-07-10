from typing import Optional

import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.BaseExtractor import BaseExtractor
from fedot_ind.core.operation.transformation.extraction.statistical import StatFeaturesExtractor


class StatsExtractor(BaseExtractor):
    """Class responsible for quantile feature generator experiment.
    Args:
        window_mode: Flag for window mode. Defaults to False.
        use_cache: Flag for cache usage. Defaults to False.
    Attributes:
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
        self.n_components = None

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

    def generate_features_from_ts(self,
                                  ts_frame: pd.DataFrame,
                                  window_length: int = None) -> pd.DataFrame:

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
        tmp_list = []
        for index, component in enumerate(ts_components):
            aggregation_df = self.extract_stats_features(component)
            tmp_list.append(aggregation_df)
        aggregation_df = pd.concat(tmp_list, axis=0)
        return aggregation_df
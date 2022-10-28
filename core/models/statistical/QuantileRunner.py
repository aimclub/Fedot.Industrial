import pandas as pd

import numpy as np

from core.models.ExperimentRunner import ExperimentRunner
from core.models.statistical.stat_features_extractor import StatFeaturesExtractor
from core.operation.utils.Decorators import time_it


class StatsRunner(ExperimentRunner):
    """
    Class responsible for quantile feature generator experiment
        :param window_mode: boolean flag that indicates whether to use window mode or not
    """

    def __init__(self,
                 window_mode: bool = False,
                 use_cache: bool = False):

        super().__init__()
        self.use_cache = use_cache
        self.ts_samples_count = None
        self.aggregator = StatFeaturesExtractor()
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.window_mode = window_mode

    def generate_features_from_ts(self, ts_frame: pd.DataFrame, window_length: int = None) -> pd.DataFrame:
        self.ts_samples_count = ts_frame.shape[0]
        self.logger.info(f'Number of TS to be processed: {self.ts_samples_count}')
        ts = self.check_for_nan(ts_frame)
        ts = pd.DataFrame(ts, dtype=float)

        if self.window_mode:
            aggregator = self.aggregator.create_baseline_features
            list_of_stat_features_on_interval = self.apply_window_for_statistical_feature(ts_data=ts,
                                                                                          feature_generator=aggregator)
            aggregation_df = pd.concat(list_of_stat_features_on_interval, axis=1)
        else:
            aggregation_df = self.aggregator.create_baseline_features(ts)

        return aggregation_df

    @time_it
    def get_features(self, ts_data, dataset_name: str = None, target: np.ndarray = None):
        return self.generate_features_from_ts(ts_data)

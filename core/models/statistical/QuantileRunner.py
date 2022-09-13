import timeit

from core.models.ExperimentRunner import ExperimentRunner
from core.models.statistical.stat_features_extractor import StatFeaturesExtractor
import pandas as pd


class StatsRunner(ExperimentRunner):
    """
    Class responsible for quantile feature generator experiment
        :param static_booster: boolean flag that indicates whether to use static booster or not
        :param window_mode: boolean flag that indicates whether to use window mode or not
    """

    def __init__(self, static_booster: bool = False,
                 window_mode: bool = False):

        super().__init__(static_booster=static_booster)

        self.ts_samples_count = None
        self.aggregator = StatFeaturesExtractor()
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.window_mode = window_mode

    def generate_features_from_ts(self, ts: pd.DataFrame, window_length: int = None) -> pd.DataFrame:
        start = timeit.default_timer()
        self.ts_samples_count = ts.shape[0]
        self.logger.info(f'Number of TS to be processed: {self.ts_samples_count}')
        ts = self.check_for_nan(ts)
        ts = pd.DataFrame(ts, dtype=float)

        if self.window_mode:
            aggregator = self.aggregator.create_baseline_features

            list_of_stat_features_on_interval = self.apply_window_for_statistical_feature(ts_data=ts,
                                                                                          feature_generator=aggregator)
            aggregation_df = pd.concat(list_of_stat_features_on_interval, axis=1)
        else:
            aggregation_df = self.aggregator.create_baseline_features(ts)

        time_elapsed = round((timeit.default_timer() - start), 2)
        self.logger.info(f'Time spent on feature generation - {time_elapsed} sec')
        return aggregation_df

    def extract_features(self, ts_data, dataset_name: str = None):
        self.logger.info('Statistical features extraction started')
        return self.generate_features_from_ts(ts_data)

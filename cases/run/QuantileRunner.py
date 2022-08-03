import timeit

from cases.run.ExperimentRunner import ExperimentRunner
from core.models.statistical.Stat_features import AggregationFeatures
from core.operation.utils.utils import *


class StatsRunner(ExperimentRunner):
    def __init__(self, static_booster: bool = False,
                 window_mode: bool = False):

        super().__init__(static_booster=static_booster)

        self.ts_samples_count = None
        self.aggregator = AggregationFeatures()
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.window_mode = window_mode

    def generate_features_from_ts(self, ts, window_length=None):
        start = timeit.default_timer()
        self.ts_samples_count = ts.shape[0]
        self.logger.info(f'Number of TS to be processed: {self.ts_samples_count}')
        ts = self.check_Nan(ts)
        ts = pd.DataFrame(ts, dtype=float)

        if self.window_mode:
            aggregator = self.aggregator.create_baseline_features

            list_with_stat_features_on_interval = apply_window_for_statistical_feature(ts_data=ts,
                                                                                       feature_generator=aggregator)
            aggregation_df = pd.concat(list_with_stat_features_on_interval, axis=1)
        else:
            aggregation_df = self.aggregator.create_baseline_features(ts)

        time_elapsed = round((timeit.default_timer() - start), 2)
        self.logger.info(f'Time spent on feature generation - {time_elapsed} sec')
        return aggregation_df

    def extract_features(self, ts_data, dataset_name: str = None):
        self.logger.info('Statistical features extraction started')
        return self.generate_features_from_ts(ts_data)

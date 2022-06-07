import pandas as pd
from sklearn.metrics import f1_score
from fedot.api.main import Fedot
from core.models.statistical.Stat_features import AggregationFeatures
from cases.run.ExperimentRunner import ExperimentRunner
from core.operation.utils.utils import *
import timeit


class StatsRunner(ExperimentRunner):
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = None,
                 static_booster: bool = False,
                 window_mode: bool = False
                 ):

        super().__init__(list_of_dataset, launches, metrics_name, fedot_params, static_booster=static_booster)
        self.aggregator = AggregationFeatures()
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.window_mode = window_mode

    def generate_features_from_ts(self, ts):
        self.ts_samples_count = ts.shape[0]
        self.logger.info(f'8 CPU on working. '
                         f'Total ts samples - {self.ts_samples_count}. '
                         f'Current sample - {self.count}')
        start = timeit.default_timer()
        ts = self.check_Nan(ts)
        ts = pd.DataFrame(ts, dtype=float)

        if self.window_mode:
            list_with_stat_features_on_interval = apply_window_for_statistical_feature(ts_data=ts,
                                                                                       feature_generator=self.aggregator.create_baseline_features)
            aggregation_df = pd.concat(list_with_stat_features_on_interval, axis=1)
        else:
            aggregation_df = self.aggregator.create_baseline_features(ts)
        self.logger.info(f'Time spent on feature generation - {timeit.default_timer() - start}')
        return aggregation_df

    def extract_features(self, ts_data):
        return self.generate_features_from_ts(ts_data)


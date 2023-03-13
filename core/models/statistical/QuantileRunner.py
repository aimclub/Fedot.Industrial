import pandas as pd

import numpy as np

from core.models.ExperimentRunner import ExperimentRunner
from core.operation.transformation.extraction.statistical import StatFeaturesExtractor
from core.architecture.abstraction.Decorators import time_it


class StatsRunner(ExperimentRunner):
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

    def __init__(self,
                 window_size: int = None,
                 window_mode: bool = False,
                 use_cache: bool = False):

        super().__init__()
        self.use_cache = use_cache
        self.aggregator = StatFeaturesExtractor()
        self.window_mode = window_mode
        self.window_size_percent = window_size
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.n_components = None

    @time_it
    def get_features(self, ts_data):

        ts_data = ts_data.fillna(method='ffill')
        try:
            single_ts_list = [pd.DataFrame(x) for x in ts_data.values]

            if single_ts_list[0].shape[0] != 1:
                single_ts_list = [x.T for x in single_ts_list]

            tmp_list = []
            for index, single_ts in enumerate(single_ts_list):
                aggregation_df = self.extract_stats_features(single_ts)
                tmp_list.append(aggregation_df)
            aggregation_df = pd.concat(tmp_list, axis=0)
            aggregation_df.reset_index(drop=True, inplace=True)

        except Exception as err:
            self.logger.debug('QuantileRunner, in method get_features error:')
            self.logger.debug(err)

            aggregation_df = self.__component_extraction(ts_data)

        return aggregation_df

    def extract_stats_features(self, single_ts):
        if self.window_mode:
            aggregator = self.aggregator.create_baseline_features
            list_of_stat_features_on_interval = self.apply_window_for_single_ts(single_ts=single_ts,
                                                                                feature_generator=aggregator,
                                                                                window_size_percent=self.window_size_percent)
            aggregation_df = pd.concat(list_of_stat_features_on_interval, axis=1)
        else:
            aggregation_df = self.aggregator.create_baseline_features(single_ts)
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


if __name__ == "__main__":
    from core.architecture.preprocessing.DatasetLoader import DataLoader

    loader = DataLoader('Car')
    train, test = loader.load_data()

    runner = StatsRunner(window_mode=True, window_size=10)

    train_feats = runner.get_features(train[0])
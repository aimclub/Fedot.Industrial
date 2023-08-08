import datetime
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from joblib import Parallel, delayed
from pandas import Index
from tqdm import tqdm

from fedot_ind.core.models.BaseExtractor import BaseExtractor


class StatsExtractor(BaseExtractor):
    """Class responsible for quantile feature generator experiment.

    Attributes:
        use_cache (bool): Flag for cache usage.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        # self.var_threshold = params.get('var_threshold')
        self.window_mode = params.get('window_mode')
        self.window_size = params.get('window_size')
        self.var_threshold = 0.1
        self.logging_params.update({'Wsize': self.window_size,
                                    'Wmode': self.window_mode,
                                    'VarTh': self.var_threshold})
        self.relevant_features = None

    def fit(self, input_data: InputData):
        pass

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """
        try:
            input_data_squeezed = np.squeeze(input_data.features, 3)
        except ValueError:
            input_data_squeezed = input_data.features
        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        v = parallel(delayed(self.generate_features_from_ts)(sample) for sample in input_data_squeezed)
        stat_features = v[0].columns
        n_components = v[0].shape[0]
        predict = self._clean_predict(np.array(v))
        # predict = self.drop_features(predict=predict,
        #                              columns=stat_features,
        #                              n_components=n_components)
        if predict.shape[1] == 36:
            print('a')
        return predict

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

    def extract_stats_features(self, ts):
        if self.window_mode:
            global_features = self.get_statistical_features(ts, add_global_features=True)
            list_of_stat_features = self.apply_window_for_stat_feature(ts_data=ts.T if ts.shape[1] == 1 else ts,
                                                                       feature_generator=self.get_statistical_features,
                                                                       window_size=self.window_size)
            aggregation_df = pd.concat(list_of_stat_features, axis=1)
            aggregation_df = pd.concat([aggregation_df, global_features], axis=1)
        else:
            statistical_features = self.get_statistical_features(ts)
            global_features = self.get_statistical_features(ts, add_global_features=True)
            aggregation_df = pd.concat([statistical_features, global_features], axis=1)
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

        tmp_list = [self.extract_stats_features(x) for x in ts_components]
        aggregation_df = pd.concat(tmp_list, axis=0)

        return aggregation_df

    def apply_window_for_stat_feature(self, ts_data: pd.DataFrame,
                                      feature_generator: callable,
                                      window_size: int = None):
        if window_size is None:
            # 10% of time series length by default
            window_size = round(ts_data.shape[1] / 10)
        else:
            window_size = round(ts_data.shape[1] * (window_size / 100))
        tmp_list = []
        window_size = max(window_size, 5)
        for i in range(0, ts_data.shape[1], window_size):
            slice_ts = ts_data.iloc[:, i:i + window_size]
            if slice_ts.shape[1] == 1:
                break
            else:
                df = feature_generator(slice_ts)
                df.columns = [x + f'_on_interval: {i} - {i + window_size}' for x in df.columns]
                tmp_list.append(df)
        return tmp_list

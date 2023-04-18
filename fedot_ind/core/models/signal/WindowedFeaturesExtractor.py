import pandas as pd

from fedot_ind.core.models.BaseExtractor import BaseExtractor


class WindowedFeatureExtractor(BaseExtractor):
    @staticmethod
    def apply_window_for_stat_feature(ts_data: pd.DataFrame,
                                      feature_generator: callable,
                                      window_size: int = None):
        ts_data = ts_data.T
        if window_size is None:
            window_size = round(ts_data.shape[1] / 10)
        else:
            window_size = round(ts_data.shape[1] / window_size)
        tmp_list = []
        for i in range(0, ts_data.shape[1], window_size):
            slice_ts = ts_data.iloc[:, i:i + window_size]
            if slice_ts.shape[1] == 1:
                break
            else:
                df = feature_generator(slice_ts)
                df.columns = [x + f'_on_interval: {i} - {i + window_size}' for x in df.columns]
                tmp_list.append(df)
        return tmp_list
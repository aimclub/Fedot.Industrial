from multiprocessing import cpu_count
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from tqdm import tqdm

from core.metrics.metrics_implementation import *
from core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation


class BaseExtractor(IndustrialCachableOperationImplementation):
    """
    Abstract class responsible for feature generators.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.current_window = None
        self.n_processes = cpu_count() // 2
        self.data_type = DataTypesEnum.table

    def fit(self, input_data: InputData):
        pass

    def _transform(self, input_data: InputData) -> np.array:
        """
            Method for feature generation for all series
        """
        v = []
        for series in tqdm(np.squeeze(input_data.features, 3)):
            v.append(self.generate_features_from_ts(series))
        predict = self._clean_predict(np.array(v))
        return predict

    @staticmethod
    def _clean_predict(predict: np.array):
        """
            Clean predict from nan, inf and reshape data for Fedot appropriate form
        """
        predict = np.where(np.isnan(predict), 0, predict)
        predict = np.where(np.isinf(predict), 0, predict)
        predict = predict.reshape(predict.shape[0], -1)
        return predict

    def generate_features_from_ts(self, ts_frame: pd.DataFrame,
                                  window_length: int = None) -> pd.DataFrame:
        """Method responsible for generation of features from time series.
        """
        pass

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

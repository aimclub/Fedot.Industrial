from multiprocessing import cpu_count
from typing import Optional
import logging

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from tqdm import tqdm

from core.metrics.metrics_implementation import *
from core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from core.operation.utils.cache import DataCacher


class BaseExtractor(IndustrialCachableOperationImplementation):
    """
    Abstract class responsible for feature generators.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.current_window = None
        self.n_processes = cpu_count() // 2
        self.data_type = DataTypesEnum.table
        self.use_cache = params.get('use_cache', False)

        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, input_data: InputData):
        pass

    def _transform(self, input_data: InputData) -> np.array:
        """
            Method for feature generation for all series
        """
        v = []
        for series in tqdm(input_data.features.values):
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
        return

    def extract_features(self, train_features: pd.DataFrame,
                         dataset_name: str = None) -> pd.DataFrame:
        """Wrapper method for feature extraction method get_features() with caching results into pickle file. The idea
        is to create a unique pointer from dataset name, subsample (test or train) and feature generator object. We
        can uniquely identify the generator in our case only using a set of parameters in the form of obj.__dict__,
        while excluding some dynamic attributes. In this way we can create a hash of incoming data unique for each
        case, and then associate it with the output data - the feature set.

        Args:
            train_features: dataframe with time series.
            dataset_name: name of dataset.

        Returns:
            Dataframe with extracted features.

        """
        generator_name = self.__class__.__name__

        if self.use_cache:
            generator_info = self.__dir__()
            data_cacher = DataCacher()
            hashed_info = data_cacher.hash_info(data=train_features,
                                                name=dataset_name,
                                                obj_info_dict=generator_info)
            try:
                return data_cacher.load_data_from_cache(hashed_info)
            except FileNotFoundError:
                self.logger.info('Cache not found. Generating features')

                features = self.generate_features_from_ts(train_features, dataset_name)
                data_cacher.cache_data(hashed_info, features)
                return features
        else:
            return self.generate_features_from_ts(train_features, dataset_name)

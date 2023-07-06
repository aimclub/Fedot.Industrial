import logging
import math
from multiprocessing import cpu_count, Pool
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from tqdm import tqdm

from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot_ind.core.operation.utils.cache import DataCacher


class BaseExtractor(IndustrialCachableOperationImplementation):
    """
    Abstract class responsible for feature generator.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        # TODO: delete this
        # self.current_window = None
        self.n_processes = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1
        self.data_type = DataTypesEnum.table
        self.use_cache = params.get('use_cache', False)

        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, input_data: InputData):
        pass

    def _transform(self, input_data: InputData) -> np.array:
        """Method for feature generation for all series

        Args:
            input_data: InputData object with features and target

        Returns:
            np.array: array with generated features

        """
        v = []
        try:
            input_data_squeezed = np.squeeze(input_data.features, 3)
        except Exception as _:
            input_data_squeezed = np.squeeze(input_data.features)

        # TODO: return to this code
        # with Pool(4) as p:
        # # with Pool(self.n_processes) as p:
        #     v = list(tqdm(p.imap(self.generate_features_from_ts, input_data_squeezed),
        #                   total=input_data.features.shape[0],
        #                   desc=f'{self.__class__.__name__} transform',
        #                   postfix=f'n_jobs:{self.n_processes}, win_size,%:{self.window_size}',
        #                   colour='green',
        #                   unit='ts',
        #                   ascii=False,
        #                   position=0,
        #                   initial=0,
        #                   leave=True)
        #              )

        for series in tqdm(input_data_squeezed,
                           total=input_data.features.shape[0],
                           desc=f'{self.__class__.__name__} transform',
                           colour='green',
                           unit='ts',
                           ascii=False,
                           position=0,
                           leave=True):
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

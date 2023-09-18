import logging
import math
from itertools import chain
from multiprocessing import cpu_count, Pool
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from joblib import Parallel, delayed
from tqdm import tqdm

from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot_ind.core.models.quantile.stat_methods import stat_methods, stat_methods_global
from fedot_ind.core.operation.caching import DataCacher


class BaseExtractor(IndustrialCachableOperationImplementation):
    """
    Abstract class responsible for feature generator.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.current_window = None
        self.n_processes = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1
        self.data_type = DataTypesEnum.table
        self.use_cache = params.get('use_cache', False)
        self.relevant_features = None
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logging_params = {'jobs': self.n_processes}

    def fit(self, input_data: InputData):
        pass

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """
        features = input_data.features
        n_samples = input_data.features.shape[0]

        try:
            input_data_squeezed = np.squeeze(features, 3)
        except Exception:
            input_data_squeezed = np.squeeze(features)

        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        feature_matrix = parallel(delayed(self.generate_features_from_ts)(sample) for sample in input_data_squeezed)
        predict = self._clean_predict(np.array([ts.features for ts in feature_matrix]))
        self.relevant_features = feature_matrix[0].supplementary_data['feature_name']
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

    def generate_features_from_ts(self, ts_frame: np.array,
                                  window_length: int = None) -> np.array:
        """Method responsible for generation of features from time series.
        """
        pass

    def extract_features(self, train_features: np.array,
                         dataset_name: str = None) -> np.array:
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

    @staticmethod
    def get_statistical_features(time_series: np.ndarray,
                                 add_global_features: bool = False) -> InputData:
        """
        Method for creating baseline quantile features for a given time series.

        Args:
            time_series: time series for which features are generated

        Returns:
            Row vector of quantile features in the form of a pandas DataFrame
            :param time_series:
            :param add_global_features:

        """
        names = []
        vals = []
        time_series = time_series.flatten()

        if add_global_features:
            list_of_methods = [*stat_methods_global.items()]
        else:
            list_of_methods = [*stat_methods.items()]

        for method in list_of_methods:
            try:
                vals.append(method[1](time_series))
                names.append(method[0])
            except ValueError:
                continue

        stat_features = InputData(idx=np.arange(len(vals)),
                                  features=np.array(vals),
                                  target='no_target',
                                  task='no_task',
                                  data_type=DataTypesEnum.table,
                                  supplementary_data={'feature_name': names})
        return stat_features

    def apply_window_for_stat_feature(self, ts_data: np.array,
                                      feature_generator: callable,
                                      window_size: int = None) -> InputData:
        if window_size is None:
            # 10% of time series length by default
            window_size = round(ts_data.shape[0] / 10)
        else:
            window_size = round(ts_data.shape[0] * (window_size / 100))
        features = []
        names = []
        window_size = max(window_size, 5)
        for i in range(0, ts_data.shape[0], window_size):
            slice_ts = ts_data[i:i + window_size]
            if slice_ts.shape[0] == 1:
                break
            else:
                stat_feature = feature_generator(slice_ts)
                features.append(stat_feature.features)
                names.append([x + f'_on_interval: {i} - {i + window_size}'
                              for x in stat_feature.supplementary_data['feature_name']])
        window_stat_features = InputData(idx=np.arange(len(features)),
                                         features=features,
                                         target='no_target',
                                         task='no_task',
                                         data_type=DataTypesEnum.table,
                                         supplementary_data={'feature_name': names})
        return window_stat_features

    @staticmethod
    def _get_feature_matrix(extraction_func: callable,
                            ts: np.array) -> InputData:
        multi_ts_stat_features = [extraction_func(x) for x in ts]
        features = np.concatenate([component.features for component in multi_ts_stat_features], axis=0)

        for index, component in enumerate(multi_ts_stat_features):
            component.supplementary_data['feature_name'] = [f'{x} for component {index}'
                                                            for x in component.supplementary_data['feature_name']]
        names = list(
            chain(*[x.supplementary_data['feature_name'] for x in multi_ts_stat_features]))

        multi_ts_stat_features = InputData(idx=np.arange(len(features)),
                                           features=features,
                                           target='no_target',
                                           task='no_task',
                                           data_type=DataTypesEnum.table,
                                           supplementary_data={'feature_name': names})

        return multi_ts_stat_features

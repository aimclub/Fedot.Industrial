import logging
import math
from itertools import chain
from multiprocessing import cpu_count

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from joblib import delayed, Parallel
from typing import Optional

from fedot_ind.api.utils.data import init_input_data
from fedot_ind.core.architecture.abstraction.decorators import convert_to_input_data
from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.repository.constanst_repository import STAT_METHODS, STAT_METHODS_GLOBAL


class BaseExtractor(IndustrialCachableOperationImplementation):
    """
    Abstract class responsible for feature generator.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.current_window = None
        self.stride = 3
        self.n_processes = math.ceil(
            cpu_count() * 0.7) if cpu_count() > 1 else 1
        self.data_type = DataTypesEnum.table
        self.use_cache = params.get(
            'use_cache', False) if params is not None else False
        self.relevant_features = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logging_params = {'jobs': self.n_processes}
        self.predict = None

    def fit(self, input_data: InputData):
        pass

    def extract_features(self, x, y) -> pd.DataFrame:
        """For those cases when you need to use feature extractor as a stangalone object
        """
        input_data = init_input_data(x, y)
        transformed_features = self.transform(
            input_data, use_cache=self.use_cache)
        try:
            return pd.DataFrame(transformed_features.predict, columns=self.relevant_features)
        except ValueError:
            return pd.DataFrame(transformed_features.predict)

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """

        parallel = Parallel(n_jobs=self.n_processes,
                            verbose=0, pre_dispatch="2*n_jobs")
        feature_matrix = parallel(delayed(self.generate_features_from_ts)(
            sample) for sample in input_data.features)

        if len(feature_matrix[0].features.shape) > 1:
            stacked_data = np.stack([ts.features for ts in feature_matrix])
            self.predict = self._clean_predict(stacked_data)
        else:
            stacked_data = np.array([ts.features for ts in feature_matrix])
            self.predict = self._clean_predict(stacked_data)
            self.predict = self.predict.reshape(self.predict.shape[0], -1)

        self.relevant_features = feature_matrix[0].supplementary_data['feature_name']
        return self.predict

    def _clean_predict(self, predict: np.array):
        """Clean predict from nan, inf and reshape data for Fedot appropriate form
        """
        predict = np.where(np.isnan(predict), 0, predict)
        predict = np.where(np.isinf(predict), 0, predict)
        return predict

    def generate_features_from_ts(self, ts_frame: np.array, window_length: int = None) -> np.array:
        """Method responsible for generation of features from time series.
        """
        pass

    @convert_to_input_data
    def get_statistical_features(self,
                                 time_series: np.ndarray,
                                 add_global_features: bool = False) -> tuple:
        """
        Method for creating baseline quantile features for a given time series.

        Args:
            add_global_features: if True, global features are added to the feature set
            time_series: time series for which features are generated

        Returns:
            InputData: object with features

        """
        names = []
        features = []
        time_series = time_series.flatten()

        if add_global_features:
            list_of_methods = [*STAT_METHODS_GLOBAL.items()]
        else:
            list_of_methods = [*STAT_METHODS.items()]

        for method in list_of_methods:
            features.append(method[1](time_series))
            names.append(method[0])
        return features, names

    @convert_to_input_data
    def apply_window_for_stat_feature(self, ts_data: np.array,
                                      feature_generator: callable,
                                      window_size: int = None) -> tuple:

        if window_size is None:
            # 10% of time series length by default
            window_size = round(ts_data.shape[0] / 10)
        else:
            window_size = round(ts_data.shape[0] * (window_size / 100))

        features = []
        names = []
        window_size = max(window_size, 5)

        if self.stride > 1:
            trajectory_transformer = HankelMatrix(time_series=ts_data,
                                                  window_size=window_size,
                                                  strides=self.stride)
            subseq_set = trajectory_transformer.trajectory_matrix
        else:
            subseq_set = np.lib.stride_tricks.sliding_window_view(ts_data,
                                                                  ts_data.shape[0] - window_size)

        for i in range(0, subseq_set.shape[1]):
            slice_ts = subseq_set[:, i]
            stat_feature = feature_generator(slice_ts)
            features.append(stat_feature.features)
            names.append([x + f'_on_interval: {i + 1} - {i + 1 + window_size}'
                          for x in stat_feature.supplementary_data['feature_name']])

        return features, names

    @convert_to_input_data
    def _get_feature_matrix(self,
                            extraction_func: callable,
                            ts: np.array) -> tuple:

        multi_ts_stat_features = [extraction_func(x) for x in ts]
        features = np.concatenate([component.features.reshape(
            1, -1) for component in multi_ts_stat_features], axis=0)

        for index, component in enumerate(multi_ts_stat_features):
            # try:
            #     component.supplementary_data['feature_name'] = [f'{x} for component {index}'
            #                                                     for x in component.supplementary_data['feature_name']]
            # except Exception as ex:
            component.supplementary_data['feature_name'] = [
                f'component {index}']
        names = list(chain(*[x.supplementary_data['feature_name']
                             for x in multi_ts_stat_features]))

        return features, names

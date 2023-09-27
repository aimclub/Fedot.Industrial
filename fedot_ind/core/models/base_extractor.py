import logging
import math
from itertools import chain
from multiprocessing import cpu_count
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from joblib import delayed, Parallel

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.models.quantile.stat_methods import stat_methods, stat_methods_global
from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix


class BaseExtractor(IndustrialCachableOperationImplementation):
    """
    Abstract class responsible for feature generator.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.current_window = None
        self.stride = None
        self.n_processes = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1
        self.data_type = DataTypesEnum.table
        self.use_cache = params.get('use_cache', False)
        self.relevant_features = None
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logging_params = {'jobs': self.n_processes}

    def fit(self, input_data: InputData):
        pass

    def extract_features(self, x, y) -> pd.DataFrame:
        """For those cases when you need to use feature extractor as a stangalone object
        """
        input_data = init_input_data(x, y)
        transformed_features = self.transform(input_data, use_cache=self.use_cache)
        try:
            return pd.DataFrame(transformed_features.predict, columns=self.relevant_features)
        except ValueError:
            return pd.DataFrame(transformed_features.predict)

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """
        features = input_data.features

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
        """Clean predict from nan, inf and reshape data for Fedot appropriate form
        """
        predict = np.where(np.isnan(predict), 0, predict)
        predict = np.where(np.isinf(predict), 0, predict)
        predict = predict.reshape(predict.shape[0], -1)
        return predict

    def generate_features_from_ts(self, ts_frame: np.array, window_length: int = None) -> np.array:
        """Method responsible for generation of features from time series.
        """
        pass

    @staticmethod
    def get_statistical_features(time_series: np.ndarray,
                                 add_global_features: bool = False) -> InputData:
        """
        Method for creating baseline quantile features for a given time series.

        Args:
            time_series: time series for which features are generated

        Returns:
            InputData: object with features

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
        self.stride = 3
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

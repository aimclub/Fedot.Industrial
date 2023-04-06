import hashlib
import logging
import os
import timeit
from multiprocessing import cpu_count
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from core.architecture.utils.utils import PROJECT_PATH
from core.metrics.metrics_implementation import *
from core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from core.operation.transformation.WindowSelection import WindowSizeSelection
from core.operation.utils.cache import DataCacher


class BaseExtractor(DataOperationImplementation):
    """Abstract class responsible for feature generators.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.current_window = None
        self.window_size = params.get('window_size')
        self.n_processes = cpu_count() // 2
        self.data_type = DataTypesEnum.table
        self.use_cache = params.get('use_cache')

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

    def get_features(self, *args, **kwargs) -> pd.DataFrame:
        """Method responsible for extracting features from time series dataframe.

        Args:
            *args: ...
            **kwargs: ...

        Returns:
            Dataframe with extracted features.
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
        if generator_name in ('StatsExtractor', 'SSAExtractor'):
            self.logger.info(f'Window mode: {self.window_mode}')

        if self.use_cache:
            generator_info = {k: v for k, v in self.__dict__.items() if k not in ['aggregator',
                                                                                  'pareto_front',
                                                                                  'spectrum_extractor']}
            hashed_info = self.hash_info(dataframe=train_features,
                                         name=dataset_name,
                                         obj_info_dict=generator_info)
            cache_path = os.path.join(PROJECT_PATH, 'cache', f'{generator_name}_' + hashed_info + '.pkl')

            try:
                self.logger.info('Trying to load features from cache')
                return self.load_features_from_cache(cache_path)
            except FileNotFoundError:
                self.logger.info('Cache not found. Generating features')
                features = self.get_features(train_features, dataset_name)
                self.save_features_to_cache(hashed_info, features)
                return features
        else:
            return self.get_features(train_features, dataset_name)

    @staticmethod
    def hash_info(dataframe: pd.DataFrame, name: str, obj_info_dict: dict) -> str:
        """Method responsible for hashing information about initial dataset, its name and feature generator.
        It utilizes md5 hashing algorithm.

        Args:
            dataframe: dataframe with time series.
            name: name of dataset.
            obj_info_dict: dictionary with information about feature generator.

        Returns:
            Hashed string.
        """
        key = (repr(dataframe) + repr(name) + repr(obj_info_dict)).encode('utf8')
        hsh = hashlib.md5(key).hexdigest()[:10]
        return hsh

    def load_features_from_cache(self, cache_path):
        start = timeit.default_timer()
        features = pd.read_pickle(cache_path)
        elapsed_time = round(timeit.default_timer() - start, 5)
        self.logger.info(f'Features loaded from cache in {elapsed_time} sec')
        return features

    def save_features_to_cache(self, hashed_data: str, features: pd.DataFrame):
        """Method responsible for saving features to cache folder. It utilizes pickle format for saving data.

        Args:
            hashed_data: hashed string.
            features: dataframe with extracted features.

        """
        cache_folder = os.path.join(PROJECT_PATH, 'cache')
        generator_name = self.__class__.__name__
        cache_file = os.path.join(PROJECT_PATH, 'cache', f'{generator_name}_' + hashed_data + '.pkl')

        os.makedirs(cache_folder, exist_ok=True)
        features.to_pickle(cache_file)
        self.logger.info(f'Features for {generator_name} cached with {hashed_data} hash')

    def generate_features_from_ts(self, ts_frame: pd.DataFrame,
                                  window_length: int = None) -> pd.DataFrame:
        """Method responsible for generation of features from time series.

        Args:
            ts_frame: dataframe with time series.
            window_length: window length for feature generation.

        Returns:
            Dataframe with extracted features.

        """
        pass

    @staticmethod
    def check_for_nan(ts: pd.DataFrame) -> pd.DataFrame:
        """Method responsible for checking if there are any NaN values in the time series dataframe
        and replacing them with 0.

        Args:
            ts: dataframe with time series.

        Returns:
            Dataframe with time series without NaN values.

        """
        if any(np.isnan(ts)):
            ts = np.nan_to_num(ts, nan=0)
        return ts

    def get_roc_auc_score(self, prediction_labels, test_labels):
        metric_roc = ROCAUC(target=test_labels, predicted_labels=prediction_labels)
        try:
            score_roc_auc = metric_roc.metric()
        except ValueError:
            self.logger.info(f'ValueError in roc_auc_score')
            score_roc_auc = 0
        return score_roc_auc

    @staticmethod
    def delete_col_by_var(dataframe: pd.DataFrame):
        for col in dataframe.columns:
            scaled_feature = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe[col].values.reshape(-1, 1))[:,
                             0]
            deviation = np.std(scaled_feature)
            if deviation < 0.05 and not col.startswith('diff'):
                del dataframe[col]
        return dataframe

    # @staticmethod
    def apply_window_for_stat_feature(self, ts_data: Union[np.ndarray, pd.DataFrame],
                                      feature_generator: callable,
                                      window_size: int = None):
        # ts_data = ts_data.T
        # ts_data = pd.DataFrame(ts_data)
        # to avoid window size bigger than half of the time series
        # self.window_size = window_size
        # # if window_size is None or (window_size > ts_data.shape[1]/2):
        # if self.current_window is None or (self.current_window > ts_data.shape[1]/2):
        #     # window size is 10% of the length of the time series
        #     self.logger.info(f'Window size not specified or > ts_length/2. Using 10% of the time series length')
        #     self.current_window = round(ts_data.shape[1] / 10)
        #
        # window = self.current_window
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

    def reduce_feature_space(self, features: pd.DataFrame,
                             var_threshold: float = 0.01,
                             corr_threshold: float = 0.98) -> pd.DataFrame:
        """Method responsible for reducing feature space.

        Args:
            features: dataframe with extracted features.
            corr_threshold: cut-off value for correlation threshold.
            var_threshold: cut-off value for variance threshold.

        Returns:
            Dataframe with reduced feature space.

        """
        init_feature_space_size = features.shape[1]

        features = self._drop_stable_features(features, var_threshold)
        features_new = self._drop_correlated_features(corr_threshold, features)

        final_feature_space_size = features_new.shape[1]

        if init_feature_space_size != final_feature_space_size:
            self.logger.info(f'Feature space reduced from {init_feature_space_size} to {final_feature_space_size}')

        return features_new

    def _drop_correlated_features(self, corr_threshold, features):
        features_corr = features.corr(method='pearson')
        mask = np.ones(features_corr.columns.size) - np.eye(features_corr.columns.size)
        df_corr = mask * features_corr
        drops = []
        for col in df_corr.columns.values:
            # continue if the feature is already in the drop list
            if np.in1d([col], drops):
                continue

            index_of_corr_feature = df_corr[abs(df_corr[col]) > corr_threshold].index
            drops = np.union1d(drops, index_of_corr_feature)

        if len(drops) == 0:
            self.logger.info('No correlated features found')
            return features

        features_new = features.copy()
        features_new.drop(drops, axis=1, inplace=True)
        return features_new

    def _drop_stable_features(self, features, var_threshold):
        try:
            variance_reducer = VarianceThreshold(threshold=var_threshold)
            variance_reducer.fit_transform(features)
            unstable_features_mask = variance_reducer.get_support()
            features = features.loc[:, unstable_features_mask]
        except ValueError:
            self.logger.info('Variance reducer has not found any features with low variance')
        return features

    def validate_window_size(self, ts: np.ndarray):
        if self.window_size is None or self.window_size > ts.shape[0] / 2:
            self.logger.info('Window size is not defined or too big (> ts_length/2)')
            self.window_size, _ = WindowSizeSelection(time_series=ts).get_window_size()
            self.logger.info(f'Window size was set to {self.window_size}')

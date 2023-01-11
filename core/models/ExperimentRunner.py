import os
from multiprocessing import cpu_count

from fedot.core.log import default_log as Logger
from sklearn.preprocessing import MinMaxScaler

from core.architecture.utils.utils import PROJECT_PATH
from core.metrics.metrics_implementation import *
from core.operation.utils.caching import DataCacher


class ExperimentRunner:
    """Abstract class responsible for feature generators.

    Args:
        feature_generator_dict: that consists of {'generator_name': generator_class} pairs.
        use_cache: flag that indicates whether to use cache or not.

    Attributes:
        current_window (int): window length for feature generation.
        logger (logging.Logger): logger instance.
        n_processes (int): number of processes for multiprocessing.
        window_mode (str): window mode for feature generation.

    """
    METRICS_NAME = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']

    def __init__(self, feature_generator_dict: dict = None,
                 use_cache: bool = False):
        self.window_mode = None
        self.use_cache = use_cache
        self.feature_generator_dict = feature_generator_dict
        self.current_window = None
        self.logger = Logger(self.__class__.__name__)
        self.n_processes = cpu_count() // 2

    def get_features(self, ts_data, ds_name) -> pd.DataFrame:
        """Method responsible for extracting features from time series dataframe.

        Args:
            ds_name: name of dataset.
            ts_data: dataframe with time series.

        Returns:
            Dataframe with extracted features.
        """
        pass

    def extract_features(self, ts_data: pd.DataFrame,
                         dataset_name: str = None) -> pd.DataFrame:
        """Wrapper method for feature extraction method get_features() with caching results into pickle file. The idea
        is to create a unique pointer from dataset name, subsample (test or train) and feature generator object. We
        can uniquely identify the generator in our case only using a set of parameters in the form of obj.__dict__,
        while excluding some dynamic attributes. In this way we can create a hash of incoming data unique for each
        case, and then associate it with the output data - the feature set.

        Args:
            ts_data: dataframe with time series.
            dataset_name: name of dataset.

        Returns:
            Dataframe with extracted features.

        """
        generator_name = self.__class__.__name__
        self.logger.info(f'Window mode: {self.window_mode}')

        if self.use_cache:
            cache_folder = os.path.join(PROJECT_PATH, 'cache')
            os.makedirs(cache_folder, exist_ok=True)
            cacher = DataCacher(data_type_prefix=f'Features of {generator_name} generator',
                                cache_folder=cache_folder)

            generator_info = self.__dir__()
            hashed_info = cacher.hash_info(dataframe=ts_data,
                                           name=dataset_name,
                                           obj_info_dict=generator_info)

            try:
                self.logger.info('Trying to load features from cache...')
                return cacher.load_data_from_cache(hashed_info=hashed_info)
            except FileNotFoundError:
                self.logger.info('Cache not found. Generating features...')
                features = self.get_features(ts_data, dataset_name)
                cacher.cache_data(hashed_info=hashed_info,
                                  data=features)
                return features
        else:
            return self.get_features(ts_data, dataset_name)

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

    @staticmethod
    def apply_window_for_stat_feature(ts_data: pd.DataFrame,
                                      feature_generator: callable,
                                      window_size: int = None):
        if window_size is None:
            window_size = round(ts_data.shape[1] / 10)
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

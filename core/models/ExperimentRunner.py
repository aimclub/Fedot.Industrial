import hashlib
import os
import timeit
from multiprocessing import cpu_count

from core.metrics.metrics_implementation import *
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import PROJECT_PATH

dict_of_dataset = dict
dict_of_win_list = dict


class ExperimentRunner:
    """Abstract class responsible for feature generators.

    Args:
        feature_generator_dict: that consists of {'generator_name': generator_class} pairs.
        use_cache: flag that indicates whether to use cache or not.

    Attributes:
        current_window (int): window length for feature generation.
        y_test (pd.DataFrame): ...
        logger (logging.Logger): logger instance.
        n_processes (int): number of processes for multiprocessing.

    """
    METRICS_NAME = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']

    def __init__(self, feature_generator_dict: dict = None,
                 use_cache: bool = False):
        self.use_cache = use_cache
        self.feature_generator_dict = feature_generator_dict
        self.current_window = None
        self.y_test = None
        self.logger = Logger().get_logger()
        self.n_processes = cpu_count() // 2

    def get_features(self, *args, **kwargs) -> pd.DataFrame:
        """Method responsible for extracting features from time series dataframe.

        Args:
            *args: ...
            **kwargs: ...

        Returns:
            pd.DataFrame: dataframe with extracted features.
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
            ts_data (pd.DataFrame): dataframe with time series.
            dataset_name (str): name of dataset.

        Returns:
            pd.DataFrame: dataframe with extracted features.

        """
        generator_name = self.__class__.__name__
        self.logger.info(f'{generator_name} is working...')
        if generator_name in ('StatsRunner', 'SSARunner'):
            self.logger.info(f'Window mode: {self.window_mode}')

        if self.use_cache:
            generator_info = {k: v for k, v in self.__dict__.items() if k not in ['aggregator']}
            hashed_info = self.hash_info(dataframe=ts_data,
                                         name=dataset_name,
                                         obj_info_dict=generator_info)
            cache_path = os.path.join(PROJECT_PATH, 'cache', f'{generator_name}_' + hashed_info + '.pkl')

            try:
                self.logger.info('Trying to load features from cache...')
                return self.load_features_from_cache(cache_path)
            except FileNotFoundError:
                self.logger.info('Cache not found. Generating features...')
                features = self.get_features(ts_data, dataset_name)
                self.save_features_to_cache(hashed_info, features)
                return features
        else:
            return self.get_features(ts_data, dataset_name)

    @staticmethod
    def hash_info(dataframe, name, obj_info_dict):
        """Method responsible for hashing information about initial dataset, its name and feature generator.
        It utilizes md5 hashing algorithm.

        Args:
            dataframe (pd.DataFrame): dataframe with time series.
            name (str): name of dataset.
            obj_info_dict (dict): dictionary with information about feature generator.

        Returns:
            str: hashed string.
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

    def save_features_to_cache(self, hashed_data, features):
        """Method responsible for saving features to cache folder. It utilizes pickle format for saving data.

        Args:
            hashed_data (str): hashed string.
            features (pd.DataFrame): dataframe with extracted features.

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
            ts_frame (pd.DataFrame): dataframe with time series.
            window_length (int): window length for feature generation.

        Returns:
            pd.DataFrame: dataframe with extracted features.

        """
        pass

    @staticmethod
    def check_for_nan(ts: pd.DataFrame) -> pd.DataFrame:
        """Method responsible for checking if there are any NaN values in the time series dataframe
        and replacing them with 0.

        Args:
            ts (pd.DataFrame): dataframe with time series.

        Returns:
            pd.DataFrame: dataframe with time series without NaN values.

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
            if dataframe[col].std() < 0.1 and not col.startswith('diff'):
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

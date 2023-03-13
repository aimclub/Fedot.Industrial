import os
from multiprocessing import cpu_count

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

from core.architecture.utils.utils import PROJECT_PATH
from core.log import default_log as logger
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
        self.logger = logger(self.__class__.__name__)
        self.n_processes = cpu_count() // 2

    def get_features(self, ts_frame: pd.DataFrame) -> pd.DataFrame:
        """Method responsible for extracting features from time series dataframe.

        Args:
            ts_frame: dataframe with time series.

        Returns:
            Dataframe with extracted features.
        """
        pass

    def extract_features(self, ts_frame: pd.DataFrame,
                         dataset_name: str = None) -> pd.DataFrame:
        """Wrapper method for feature extraction method get_features() with caching results into pickle file. The idea
        is to create a unique pointer from dataset name, subsample (test or train) and feature generator object. We
        can uniquely identify the generator in our case only using a set of parameters in the form of obj.__dict__,
        while excluding some dynamic attributes. In this way we can create a hash of incoming data unique for each
        case, and then associate it with the output data - the feature set.

        Args:
            ts_frame: dataframe with time series.
            dataset_name: name of dataset.

        Returns:
            Dataframe with extracted features.

        """
        generator_name = self.__class__.__name__
        self.logger.info(f'Window mode: {self.window_mode}')

        if self.use_cache:
            cache_folder = os.path.join(PROJECT_PATH, 'cache')
            os.makedirs(cache_folder, exist_ok=True)
            cacher = DataCacher(data_type=f'Features',
                                cache_folder=cache_folder,
                                object_name=generator_name)

            generator_info = self.__dir__()
            hashed_info = cacher.hash_info(dataframe=ts_frame,
                                           name=dataset_name,
                                           obj_info_dict=generator_info,
                                           generator_name=generator_name)

            try:
                self.logger.info('Trying to load features from cache...')
                return cacher.load_data_from_cache(hashed_info=hashed_info)
            except FileNotFoundError:
                self.logger.info('Cache not found. Generating features...')
                features = self.get_features(ts_frame=ts_frame)
                cacher.cache_data(hashed_info=hashed_info,
                                  data=features)
                return features
        else:
            return self.get_features(ts_frame=ts_frame)

    @staticmethod
    def check_for_nan(single_ts: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Method responsible for checking if there are any NaN values in the time series dataframe
        and replacing them with 0.

        Args:
            single_ts: dataframe with time series.

        Returns:
            Dataframe with time series without NaN values.

        """
        if any(np.isnan(single_ts)):
            single_ts = np.nan_to_num(single_ts, nan=0)
        return single_ts

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

    def reduce_feature_space(self, features: pd.DataFrame,
                             variance_thr: float = 0.01,
                             corr_thr: float = 0.96) -> pd.DataFrame:
        """(Unfinished yet) Method responsible for reducing feature space.

        Args:
            features: dataframe with features.

        Returns:
            Dataframe with reduced feature space.

        """

        quazi_constant_filter = VarianceThreshold(threshold=variance_thr)

        quazi_constant_filter.fit_transform(features)
        constant_columns = quazi_constant_filter.get_support(indices=True)
        features = features.iloc[:, constant_columns]

        corr_matrix = features.corr().abs()
        upped = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upped.columns if any(upped[column] > corr_thr)]
        features = features.drop(features[to_drop], axis=1)

        return features


    @staticmethod
    def apply_window_for_single_ts(single_ts: pd.DataFrame,
                                   feature_generator: callable,
                                   window_size_percent: int = None):

        # define window size which is percent of the whole time series
        if window_size_percent is None:
            window_size = round(single_ts.shape[1] / 10)
        else:
            window_size = round(single_ts.shape[1] * window_size_percent / 100)

        tmp_list = []

        for i in range(0, single_ts.shape[1], window_size):
            slice_ts = single_ts.iloc[:, i:i + window_size]

            if slice_ts.shape[1] == 1:
                break
            else:
                df = feature_generator(slice_ts)
                df.columns = [x + f'_on_interval: {i} - {i + window_size}' for x in df.columns]
                tmp_list.append(df)
        return tmp_list

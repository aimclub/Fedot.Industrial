from core.metrics.metrics_implementation import *
from core.operation.utils.Decorators import exception_decorator
from core.operation.utils.LoggerSingleton import Logger

dict_of_dataset = dict
dict_of_win_list = dict


class ExperimentRunner:
    """
    Abstract class responsible for feature generators
        :param feature_generator_dict: dict that consists of 'generator_name': generator_class pairs
        :param boost_mode: defines whether to use error correction model or not
        :param static_booster: defines whether to use static booster or dynamic
    """
    METRICS_NAME = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']

    def __init__(self, feature_generator_dict: dict = None,
                 boost_mode: bool = True,
                 static_booster: bool = False):
        self.feature_generator_dict = feature_generator_dict
        self.count = 0
        self.window_length = None
        self.y_test = None
        self.logger = Logger().get_logger()
        self.boost_mode = boost_mode
        self.static_booster = static_booster

    def generate_features_from_ts(self, ts_frame: pd.DataFrame,
                                  window_length: int = None) -> pd.DataFrame:
        """
        Method responsible for generation of features from time series.

        :return: dataframe with generated features
        """
        pass

    def extract_features(self, ts_data: pd.DataFrame, dataset_name: str = None):
        """
        Method responsible for extracting features from time series dataframe
        :param ts_data: dataframe with time series data
        :param dataset_name:
        :return: pd.DataFrame with extracted features
        """
        pass

    @staticmethod
    def check_for_nan(ts: pd.DataFrame) -> pd.DataFrame:
        """
        Method responsible for checking if there are any NaN values in the time series dataframe
        and replacing them with 0
        :param ts: dataframe with time series data
        :return: dataframe with time series data without NaN values
        """
        if any(np.isnan(ts)):
            ts = np.nan_to_num(ts, nan=0)
        return ts

    @exception_decorator(exception_return=0.5)
    def get_roc_auc_score(self, prediction, test_data):
        metric_roc = ROCAUC(target=test_data, predicted_labels=prediction.predict)
        try:
            score_roc_auc = metric_roc.metric()
        except ValueError:
            self.logger.info(f'ValueError in roc_auc_score')
            score_roc_auc = 0
        return score_roc_auc

    @staticmethod
    def delete_col_by_var(dataframe: pd.DataFrame):
        for col in dataframe.columns:
            if dataframe[col].var() < 0.001 and not col.startswith('diff'):
                del dataframe[col]
        return dataframe

    @staticmethod
    def apply_window_for_statistical_feature(ts_data: pd.DataFrame,
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

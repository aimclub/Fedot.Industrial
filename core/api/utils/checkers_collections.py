from typing import Union

import numpy as np
import pandas as pd

from core.operation.utils.LoggerSingleton import Logger


class ParameterCheck:
    def __init__(self):
        self.logger = Logger().get_logger()

    def check_window_sizes(self, config_dict: dict,
                           dataset_name: str,
                           train_data: Union[pd.DataFrame, np.ndarray]):

        for key in config_dict['feature_generator_params'].keys():
            if key.startswith('spectral') or 'spectral' in key:
                self.logger.info(f'CHECK WINDOW SIZES FOR DATASET-{dataset_name} AND {key} method')
                if dataset_name not in config_dict['feature_generator_params'][key].keys():
                    ts_length = train_data[0].shape[1]
                    list_of_WS = list(map(lambda x: round(ts_length / x), [10, 5, 3]))
                    config_dict['feature_generator_params'][key]['window_sizes'][dataset_name] = list_of_WS
                    self.logger.info(f'THERE ARE NO PREDEFINED WINDOWS. '
                                     f'DEFAULTS WINDOWS SIZES WAS SET - {list_of_WS}. '
                                     f'THATS EQUAL 10/20/30% OF TS LENGTH')
        return config_dict

    def check_metric_type(self, n_classes):
        if n_classes > 2:
            self.logger.info('Metric for evaluation - F1')
            return 'f1'
        else:
            self.logger.info('Metric for evaluation - ROC-AUC')
            return 'roc_auc'


class DataCheck:
    def __init__(self, logger: Logger):
        self.logger = logger

    @staticmethod
    def _replace_inf_with_nans(input_data: np.ndarray):
        values_to_replace = [np.inf, -np.inf]
        features_with_replaced_inf = np.where(np.isin(input_data,
                                                      values_to_replace),
                                              np.nan,
                                              input_data)
        input_data = features_with_replaced_inf
        return input_data

    def _check_for_nan(self, input_data: np.ndarray) -> np.ndarray:
        """
        Method responsible for checking if there are any NaN values in the time series dataframe
        and replacing them with 0

        :param input_data: data with NaN values
        :return:data without NaN values
        """
        if np.any(np.isnan(input_data)):
            input_data = np.nan_to_num(input_data, nan=0)
        return input_data

    def check_data(self, input_data: pd.DataFrame, return_df: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        if type(input_data) != pd.DataFrame:
            return input_data
        else:
            filled_data = input_data.apply(lambda x: self._replace_inf_with_nans(x))
            filled_data = filled_data.apply(lambda x: self._check_for_nan(x))
            if filled_data.shape[0] == input_data.shape[0] and filled_data.shape[1] == input_data.shape[1]:
                input_data = filled_data

            if return_df:
                return input_data
            else:
                return input_data.values

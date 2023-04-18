import logging
from typing import Union

import numpy as np
import pandas as pd


class ParameterCheck:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_window_sizes(self, config_dict: dict,
                           dataset_name: str,
                           train_data: Union[pd.DataFrame, np.ndarray]):

        for generator in config_dict['feature_generator']:
            if generator in ['spectral', 'window_spectral']:
                self.logger.info(f'Check window sizes for {generator} generator and {dataset_name} dataset')
                if dataset_name not in config_dict['feature_generator_params'][generator]['window_sizes'].keys():

                    ts_length = train_data[0].shape[1]
                    window_sizes_list = list(map(lambda x: round(ts_length / x), [10, 5, 3]))
                    config_dict['feature_generator_params'][generator]['window_sizes'][dataset_name] = window_sizes_list
                    self.logger.info(f'Window sizes for {dataset_name} are not specified. '
                                     f'Auto-selected: {window_sizes_list}')
                else:
                    self.logger.info(f'Window sizes for dataset {dataset_name} are predefined')
            else:
                continue

        return config_dict

    def check_metric_type(self, target):
        n_classes = np.unique(target).shape[0]
        if n_classes > 2:
            self.logger.info('Metric for evaluation - F1')
            return 'f1'
        else:
            self.logger.info('Metric for evaluation - ROC-AUC')
            return 'roc_auc'

    def check_baseline_type(self, config_dict: None, model_params: dict):
        if config_dict is not None:
            baseline_type = config_dict['baseline']
        elif 'baseline_type' in model_params.keys():
            baseline_type = model_params['baseline']
        else:
            baseline_type = None
        return baseline_type

    def get_ds_and_generator_combinations(self, datasets, generators):
        combinations = []
        for dataset in datasets:
            for generator in generators:
                combinations.append((dataset, generator))
        return combinations


class DataCheck:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

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
        """Method responsible for checking if there are any NaN values in the time series dataframe
        and replacing them with 0

        Args:
            input_data: time series dataframe with NaN values

        Returns:
            input_data: time series dataframe without NaN values

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
            else:
                self.logger.error('Encontered error during extracted features checking')
                raise ValueError('Data contains NaN values')

            if return_df:
                return input_data
            else:
                return input_data.values

import logging
from typing import Union

import numpy as np
import pandas as pd


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

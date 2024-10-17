from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.preprocessing import LabelEncoder

from fedot_ind.core.operation.transformation.splitter import TSTransformer
from fedot_ind.tools.synthetic.anomaly_generator import AnomalyGenerator
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


def check_multivariate_data(data: pd.DataFrame) -> tuple:
    """
    Checks if the provided pandas DataFrame contains multivariate data.

    Args:
        data (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
        bool: True if the DataFrame contains multivariate data (nested columns), False otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        return len(data.shape) > 2, data
    else:
        return isinstance(data.iloc[0, 0], pd.Series), data.values


def init_input_data(X: pd.DataFrame,
                    y: Optional[np.ndarray],
                    task: str = 'classification') -> InputData:
    """
    Initializes a Fedot InputData object from input features and target.

    Args:
        X: The DataFrame containing features.
        y: The NumPy array containing target values.
        task: The machine learning task type ("classification" or "regression"). Defaults to "classification".

    Returns:
        InputData: The initialized Fedot InputData object.

    """

    is_multivariate_data, features = check_multivariate_data(X)
    task_dict = {'classification': Task(TaskTypesEnum.classification),
                 'regression': Task(TaskTypesEnum.regression)}

    if y is not None and isinstance(
            y[0], np.str_) and task == 'classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    elif y is not None and isinstance(y[0], np.str_) and task == 'regression':
        y = y.astype(float)

    data_type = DataTypesEnum.image if is_multivariate_data else DataTypesEnum.table
    input_data = InputData(idx=np.arange(len(X)),
                           features=np.array(features.tolist()).astype(float),
                           target=y.reshape(-1, 1) if y is not None else y,
                           task=task_dict[task],
                           data_type=data_type)

    if input_data.target is not None:
        if task == 'regression':
            input_data.target = input_data.target.squeeze()
        elif task == 'classification':
            input_data.target[input_data.target == -1] = 0

    # Replace NaN and infinite values with 0 in features
    input_data.features = np.where(
        np.isnan(input_data.features), 0, input_data.features)
    input_data.features = np.where(
        np.isinf(input_data.features), 0, input_data.features)

    return input_data


class SynthTimeSeriesData:
    def __init__(self, config: dict):
        """
        Args:
            config: dict with config for synthetic ts_data.
        """
        self.config = config

    def generate_ts(self, ):
        """
        Method to generate synthetic time series

        Returns:
            synthetic time series data.

        """
        return TimeSeriesGenerator(self.config).get_ts()

    def generate_anomaly_ts(self,
                            ts_data,
                            plot: bool = False,
                            overlap: float = 0.1):
        """
        Method to generate anomaly time series

        Args:
            ts_data: either np.ndarray or dict with config for synthetic ts_data.
            overlap: float, ``default=0.1``. Defines the maximum overlap between anomalies.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.

        Returns:
            returns initial time series data, modified time series data and dict with anomaly intervals.

        """

        generator = AnomalyGenerator(config=self.config)
        init_synth_ts, mod_synth_ts, synth_inters = generator.generate(time_series_data=ts_data,
                                                                       plot=plot, overlap=overlap)

        return init_synth_ts, mod_synth_ts, synth_inters

    def split_ts(self,
                 time_series,
                 binarize: bool = False,
                 plot: bool = True) -> tuple:
        """
        Method to split time series with anomalies into features and target.

        Args:
            time_series (npp.array):
            binarize: if True, target will be binarized. Recommended for classification task if classes are imbalanced.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.

        Returns:
            features (pd.DataFrame) and target (np.array).

        """

        features, target = TSTransformer().transform_for_fit(
            plot=plot, binarize=binarize, series=time_series, anomaly_dict=self.config)
        return features, target

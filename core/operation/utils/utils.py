import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PROJECT_PATH = Path(__file__).parent.parent.parent.parent


def save_results(predictions: Union[np.ndarray, pd.DataFrame],
                 predictions_proba: Union[np.ndarray, pd.DataFrame],
                 target: Union[np.ndarray, pd.Series],
                 path_to_save: str,
                 inference: float,
                 fit_time: float,
                 window: int,
                 metrics: dict = None):

    path_results = os.path.join(path_to_save, 'test_results')
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    if type(predictions_proba) is not pd.DataFrame:
        df_preds = pd.DataFrame(predictions_proba)
        df_preds['Target'] = target
        df_preds['Preds'] = predictions
    else:
        df_preds = predictions_proba
        df_preds['Target'] = target.values

    if type(metrics) is str:
        df_metrics = pd.DataFrame()
    else:
        df_metrics = pd.DataFrame.from_records(data=[x for x in metrics.items()]).reset_index()
    df_metrics['Inference'] = inference
    df_metrics['Fit_time'] = fit_time
    df_metrics['window'] = window

    for p, d in zip(['probs_preds_target.csv', 'metrics.csv'],
                    [df_preds, df_metrics]):
        full_path = os.path.join(path_results, p)
        d.to_csv(full_path)
    return


def path_to_save_results() -> str:
    path = PROJECT_PATH
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path


def fill_by_mean(column: str, feature_data: pd.DataFrame):
    feature_data.fillna(value=feature_data[column].mean(), inplace=True)


def read_tsv(file_name: str):
    """
    Read tsv file that contains data for classification experiment. Data must be placed
    in 'data' folder with .tsv extension.
    :param file_name:
    :return: (x_train, x_test) - pandas dataframes and (y_train, y_test) - numpy arrays
    """
    df_train = pd.read_csv(
        os.path.join(PROJECT_PATH, 'data', file_name, f'{file_name}_TRAIN.tsv'),
        sep='\t',
        header=None)

    x_train = df_train.iloc[:, 1:]
    y_train = df_train[0].values

    df_test = pd.read_csv(
        os.path.join(PROJECT_PATH, 'data', file_name, f'{file_name}_TEST.tsv'),
        sep='\t',
        header=None)

    x_test = df_test.iloc[:, 1:]
    y_test = df_test[0].values

    return (x_train, x_test), (y_train, y_test)

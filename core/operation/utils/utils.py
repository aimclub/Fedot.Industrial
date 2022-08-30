from multiprocessing import Pool
from typing import Union

import os
import numpy as np
import pandas as pd


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


def project_path() -> str:
    abs_path = os.path.abspath(os.path.curdir)
    abs_path = os.path.dirname(abs_path)
    return abs_path


def path_to_save_results() -> str:
    path = project_path()
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path


def delete_col_by_var(dataframe: pd.DataFrame):
    for col in dataframe.columns:
        if dataframe[col].var() < 0.001 and not col.startswith('diff'):
            del dataframe[col]
    return dataframe


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


def fill_by_mean(column: str, feature_data: pd.DataFrame):
    feature_data.fillna(value=feature_data[column].mean(), inplace=True)


def threading_operation(ts_frame: pd.DataFrame,
                        function_for_feature_extraction: callable):
    pool = Pool(8)
    features = pool.map(function_for_feature_extraction, ts_frame)
    pool.close()
    pool.join()
    return features

import os
from typing import Union
import numpy as np
import pandas as pd


def save_results(predictions: Union[np.ndarray, pd.DataFrame],
                 prediction_proba: Union[np.ndarray, pd.DataFrame],
                 target: Union[np.ndarray, pd.Series],
                 path_to_save: str,
                 inference: float,
                 fit_time: float,
                 metrics: dict = None):
    path_results = os.path.join(path_to_save, 'test_results')
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    if type(prediction_proba) is not pd.DataFrame:
        df_preds = pd.DataFrame(prediction_proba)
        df_preds['Target'] = target
        df_preds['Preds'] = predictions
    else:
        df_preds = prediction_proba
        df_preds['Target'] = target.values

    if type(metrics) is str:
        df_metrics = pd.DataFrame()
    else:
        df_metrics = pd.DataFrame.from_records(data=[x for x in metrics.items()]).reset_index()
    df_metrics['Inference'] = inference
    df_metrics['Fit_time'] = fit_time
    for p, d in zip(['probs_preds_target.csv', 'metrics.csv'],
                    [df_preds, df_metrics]):
        full_path = os.path.join(path_results, p)
        d.to_csv(full_path)

    return


def project_path() -> str:
    name_project = 'IndustrialTS'
    abs_path = os.path.abspath(os.path.curdir)
    while os.path.basename(abs_path) != name_project:
        abs_path = os.path.dirname(abs_path)
    return abs_path


def path_to_save_results() -> str:
    path = project_path()
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path

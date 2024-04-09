import random
from pathlib import Path
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.metrics.metrics_implementation import calculate_forecasting_metric

from fedot_ind.tools.loader import DataLoader

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.architecture.settings.computational import backend_methods as np

ts_datasets = {
    'm4_yearly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Yearly.csv'),
    'm4_weekly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Weekly.csv'),
    'm4_daily': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Daily.csv'),
    'm4_monthly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Monthly.csv'),
    'm4_quarterly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Quarterly.csv')}


def evaluate_metric(target, prediction):
    try:
        if len(np.unique(target)) > 2:
            metric = f1_score(target, prediction, average='weighted')
        else:
            metric = roc_auc_score(target, prediction, average='weighted')
    except Exception:
        metric = f1_score(target, np.argmax(
            prediction, axis=1), average='weighted')
    return metric


def compare_forecast_with_sota(dataset_name, horizon):
    autogluon = PROJECT_PATH + f'/benchmark/results/benchmark_results/autogluon/' \
                               f'{dataset_name}_{horizon}_forecast_vs_actual.csv'
    n_beats = PROJECT_PATH + f'/benchmark/results/benchmark_results/nbeats/' \
                             f'{dataset_name}_{horizon}_forecast_vs_actual.csv'
    n_beats = pd.read_csv(n_beats)
    autogluon = pd.read_csv(autogluon)

    n_beats_forecast = calculate_forecasting_metric(target=n_beats['value'].values,
                                                    labels=n_beats['predict'].values)
    autogluon_forecast = calculate_forecasting_metric(target=autogluon['value'].values,
                                                      labels=autogluon['predict'].values)
    return n_beats['predict'].values,n_beats_forecast, autogluon['predict'].values, autogluon_forecast


def industrial_forecasting_modelling_loop(dataset_name: str = None,
                                          benchmark: str = None,
                                          horizon: int = 1,
                                          finetune: bool = False,
                                          api_config: dict = None):
    industrial = FedotIndustrial(**api_config)
    train_data, _ = DataLoader(dataset_name=dataset_name).load_forecast_data(folder=benchmark)
    target = train_data.values[-horizon:].flatten()
    if finetune:
        model = industrial.finetune(train_data)
    else:
        model = industrial.fit(train_data)

    labels = industrial.predict(train_data)
    metrics = industrial.get_metrics(target=target,
                                     metric_names=('smape', 'rmse', 'median_absolute_error'))
    return model, labels, metrics, target


def industrial_common_modelling_loop(dataset_name: str = None,
                                     finetune: bool = False,
                                     api_config: dict = None,
                                     metric_names: tuple = ('r2', 'rmse', 'mae')):
    industrial = FedotIndustrial(**api_config)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    if finetune:
        model = industrial.finetune(train_data)
    else:
        model = industrial.fit(train_data)

    labels = industrial.predict(test_data)
    metrics = industrial.get_metrics(target=test_data[1],
                                     rounding_order=3,
                                     metric_names=metric_names)
    return model, labels, metrics

import random
from pathlib import Path

import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
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

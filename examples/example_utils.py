import random
from pathlib import Path

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from sklearn.metrics import f1_score, roc_auc_score
from fedot_ind.api.utils.path_lib import PROJECT_PATH

ts_datasets = {
    'm4_yearly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4YearlyTest.csv'),
    'm4_weekly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4WeeklyTest.csv'),
    'm4_daily': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4DailyTest.csv'),
    'm4_monthly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4MonthlyTest.csv'),
    'm4_quarterly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4QuarterlyTest.csv')}


def check_multivariate_data(data: pd.DataFrame) -> bool:
    if isinstance(data.iloc[0, 0], pd.Series):
        return True
    else:
        return False


def evaluate_metric(target, prediction):
    try:
        if len(np.unique(target)) > 2:
            metric = f1_score(target, prediction, average='weighted')
        else:
            metric = roc_auc_score(target, prediction, average='weighted')
    except Exception:
        metric = 0
    return metric


def init_input_data(X: pd.DataFrame, y: np.ndarray) -> InputData:
    is_multivariate_data = check_multivariate_data(X)
    if is_multivariate_data:
        input_data = InputData(idx=np.arange(len(X)),
                               features=np.array(X.values.tolist()),
                               target=y.reshape(-1, 1),
                               task=Task(TaskTypesEnum.classification),
                               data_type=DataTypesEnum.image)
    else:
        input_data = InputData(idx=np.arange(len(X)),
                               features=X.values,
                               target=np.ravel(y).reshape(-1, 1),
                               task=Task(TaskTypesEnum.classification),
                               data_type=DataTypesEnum.table)
    return input_data


def get_ts_data(dataset='m4_monthly', horizon: int = 30, m4_id=None):
    time_series = pd.read_csv(ts_datasets[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if not m4_id:
        label = random.choice(np.unique(time_series['label']))
    else:
        label = m4_id
    print(label)
    time_series = time_series[time_series['label'] == label]

    if dataset not in ['australia']:
        idx = pd.to_datetime(time_series['idx'].values)
    else:
        # non datetime indexes
        idx = time_series['idx'].values

    time_series = time_series['value'].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input)
    return train_data, test_data, label

# def visualise_and_save():
#     for class_number in np.unique(train_data[1]):
#         for basis_name, basis in zip(['basis_before_power_iterations', 'basis_after_power_iterations'],
#                                      [basis_1d_raw, basis_1d_approx]):
#             class_idx = np.where(train_data[1] == class_number)[0]
#             class_slice = np.take(basis, class_idx, 0)
#             pd.DataFrame(np.median(class_slice, axis=0)).T.plot()
#             # plt.show()
#             plt.savefig(f'{dataset_name}/{basis_name}_{class_number}_median_component.png', bbox_inches='tight')
#             # plt.title(f'mean_{basis_name}_components_for_{class_number}_class')
#     rank_distrib = pd.DataFrame([rank_distribution_befor, rank_distribution_after]).T
#     rank_distrib.columns = ['HT_approach',
#                             'Proposed_approach']
#     rank_distrib.plot(kind='kde')
#     # plt.show()
#     rank_dispersion_ht = np.round(rank_distrib['HT_approach'].std(), 3)
#     rank_dispersion_new = np.round(rank_distrib['Proposed_approach'].std(), 3)
#     plt.savefig(f'{dataset_name}/rank_distrib. '
#                 f'Classical_rank_{low_rank_befor}_std_{rank_dispersion_ht}.'
#                 f'New_{low_rank_after}_std_{rank_dispersion_new}.png', bbox_inches='tight')
#     rank_distrib['classes'] = train_data[1]
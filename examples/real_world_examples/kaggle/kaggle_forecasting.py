import csv
from copy import deepcopy

import numpy as np
import pandas as pd

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

finetune = False
warehouse_to_item_id = {
    'Prague_1': 1,
    'Brno_1': 2,
    'Prague_2': 3,
    'Prague_3': 4,
    'Budapest_1': 5,
    'Munich_1': 6,
    'Frankfurt_1': 7
}
item_id_to_warehouse = {v: k for k, v in warehouse_to_item_id.items()}
cat = ['holiday',
       'shutdown', 'mini_shutdown', 'shops_closed', 'winter_school_holidays',
       'school_holidays', 'blackout', 'mov_change', 'frankfurt_shutdown',
       'precipitation', 'snow']


def forecasting_loop(dataset_dict):
    metric_names = ('rmse', 'smape')
    ts_dict = {}
    for time_series_id in range(1, 8):
        horizon = len(dataset_dict['test_data'][dataset_dict['test_data']['warehouse']
                                                == item_id_to_warehouse[time_series_id]])
        copy_data = deepcopy(dataset_dict)
        features = copy_data['train_data'][copy_data['train_data']['item_id']
                                           == time_series_id]['target'].values
        copy_data['train_data'] = (features, features)
        target, historical_data = prepare_target(dataset_dict['train_data'], time_series_id)
        copy_data['test_data'] = (features, target[-horizon:])
        forecast_params = {'forecast_length': horizon}
        api_config = dict(
            problem='ts_forecasting',
            metric='rmse',
            timeout=5,
            with_tuning=False,
            industrial_strategy='forecasting_assumptions',
            industrial_strategy_params={
                'industrial_task': 'ts_forecasting',
                'data_type': 'time_series'},
            task_params=forecast_params,
            logging_level=50)
        result_dict = ApiTemplate(api_config=api_config,
                                  metric_list=metric_names).eval(dataset=copy_data,
                                                                 finetune=finetune)
        ts_dict.update({time_series_id: result_dict})

    return ts_dict


def load_kaggle_data(path_train: str = './data/train.csv', path_test: str = './data/test.csv'):
    train = (
        pd.read_csv(
            path_train,
            parse_dates=["date"]
        ).rename(
            columns={"orders": "target", "date": "timestamp", "id": "item_id"}
        )
        .set_index("timestamp")
        # .resample("24H", origin="end", label="left", closed="right")
        # .mean()
        .reset_index()
    )

    # load the test set
    test = pd.read_csv(path_test, parse_dates=["date"])
    train['item_id'] = train['warehouse'].map(warehouse_to_item_id)
    train[cat] = train[cat].astype('category')
    train = train.drop(columns=['holiday_name'])
    return train, test


def prepare_target(train, time_series_id):
    historical_data = {
        'user_activity_1': np.ravel(np.array(train[train['item_id'] == time_series_id]['user_activity_1'])),
        'user_activity_2': np.ravel(np.array(train[train['item_id'] == time_series_id]['user_activity_2'])),
        'target': np.ravel(np.array(train[train['item_id'] == time_series_id]['target'])),
    }

    target = np.ravel(np.array(train[train['item_id'] == time_series_id]['target']))
    return target, historical_data


def save_predict():
    columns = ['id', 'orders']
    rows = [columns] + list(zip(test['id'], test['orders']))
    with open('submission.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == "__main__":
    train, test = load_kaggle_data()
    dataset_dict = dict(train_data=train, test_data=test)
    result_dict = forecasting_loop(dataset_dict)
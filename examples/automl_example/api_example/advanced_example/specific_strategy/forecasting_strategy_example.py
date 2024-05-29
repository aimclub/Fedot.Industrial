from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader

dataset_name = 'Lightning7'
metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
api_config = dict(problem='ts_forecasting',
                  metric='rmse',
                  timeout=15,
                  with_tuning=False,
                  industrial_strategy='forecasting_assumptions',
                  industrial_strategy_params={'industrial_task': 'ts_forecasting',
                                              'data_type': 'time_series'},
                  logging_level=20)
train_data, test_data = DataLoader(dataset_name).load_data()
industrial = FedotIndustrial(**api_config)
industrial.fit(train_data)
predict = industrial.predict(test_data)
_ = 1

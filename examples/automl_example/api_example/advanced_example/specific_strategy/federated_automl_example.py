from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader

dataset_name = 'Lightning7'
metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
api_config = dict(problem='classification',
                  metric='f1',
                  timeout=5,
                  n_jobs=2,
                  industrial_strategy='federated_automl',
                  industrial_strategy_params={},
                  logging_level=20)
train_data, test_data = DataLoader(dataset_name).load_data()
industrial = FedotIndustrial(**api_config)
industrial.fit(train_data)
predict = industrial.predict(test_data)
_ = 1

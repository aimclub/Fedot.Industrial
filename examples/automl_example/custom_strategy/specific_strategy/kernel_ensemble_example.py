from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader

dataset_name = 'Lightning7'
metric_names = ('f1', 'accuracy')
api_config = dict(
    problem='classification',
    metric='f1',
    timeout=5,
    n_jobs=2,
    with_tuning=False,
    industrial_strategy='kernel_automl',
    industrial_strategy_params={
        'industrial_task': 'classification',
        'data_type': 'tensor',
        'learning_strategy': 'all_classes',
        'head_model': 'rf'
    },
    logging_level=20)

if __name__ == "__main__":
    train_data, test_data = DataLoader(dataset_name).load_data()
    industrial = FedotIndustrial(**api_config)
    industrial.fit(train_data)
    predict = industrial.predict(test_data, 'ensemble')
    predict_proba = industrial.predict_proba(test_data, 'ensemble')
    metric = industrial.get_metrics(target=test_data[1],
                                    metric_names=metric_names)

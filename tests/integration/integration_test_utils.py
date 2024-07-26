from fedot_ind.tools.loader import DataLoader
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


METRICS = {
    'classification': 'f1',
    'regression': 'mse',
    'ts_forecasting': 'mse'
}

# def data(name):
#     train_data, test_data = DataLoader(dataset_name=name).load_data()
#     return train_data, test_data

def data(type_, task='classification'):
    return TimeSeriesDatasetsGenerator(num_samples=100,
                                        task=task,
                                        max_ts_len=50,
                                        binary=True,
                                        test_size=0.5,
                                        multivariate=(type_ == 'multivariate')).generate_data()


def launch_api(problem, industrial_strategy, train_data, test_data):
    api_config = dict(problem=problem,
                  metric=METRICS[problem],
                  timeout=0.1,
                  n_jobs=-1,
                  industrial_strategy=industrial_strategy,
                  industrial_task_params={'industrial_task': problem,
                                          'data_type': 'time_series'},
                  use_input_preprocessing=True,
                  industrial_strategy_params={},
                  logging_level=20)
    industrial = FedotIndustrial(**api_config)

    industrial.fit(train_data)
    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    assert labels is not None
    assert probs is not None
    assert probs.min() >= 0
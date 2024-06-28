from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def multi_data():
    train_data, test_data = DataLoader(dataset_name='Epilepsy').load_data()
    return train_data, test_data


def uni_data():
    train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
    return train_data, test_data


def combinations(data, strategy):
    return [[d, s] for d in data for s in strategy]


def test_federated_clf():
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=5,
                      n_jobs=2,
                      industrial_strategy='federated_automl',
                      industrial_strategy_params={},
                      logging_level=20)

    # Huge synthetic dataset for experiment
    train_data, test_data = TimeSeriesDatasetsGenerator(num_samples=1800,
                                                        task='classification',
                                                        max_ts_len=50,
                                                        binary=True,
                                                        test_size=0.5,
                                                        multivariate=False).generate_data()

    industrial = FedotIndustrial(**api_config)
    industrial.fit(train_data)
    predict = industrial.predict(test_data)

    assert predict is not None


# ['federated_automl',
#  'kernel_automl',]

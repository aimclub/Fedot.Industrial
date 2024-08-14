from our_approach.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator
from our_approach.api.main import MainClass


def classification_data():
    generator = TimeSeriesDatasetsGenerator(task='classification',
                                            binary=True,
                                            multivariate=False)
    train_data, test_data = generator.generate_data()

    return train_data, test_data


def kernel_ensemble():
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=0.1,
                      n_jobs=1,
                      industrial_strategy='kernel_automl',
                      industrial_strategy_params={},
                      logging_level=60)
    industrial = MainClass(**api_config)
    train_data, test_data = classification_data()
    industrial.fit(train_data)
    predict = industrial.predict(test_data)

    assert predict is not None

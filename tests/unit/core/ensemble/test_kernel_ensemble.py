from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.data import init_input_data


def classification_data():
    generator = TimeSeriesDatasetsGenerator(task='classification',
                                            binary=True,
                                            multivariate=False)
    train_data, test_data = generator.generate_data()

    # train_input_data = init_input_data(
    #     train_data[0],
    #     train_data[1],
    #     task='classification')
    #
    # test_input_data = init_input_data(
    #     test_data[0],
    #     test_data[1],
    #     task='classification')

    # return train_input_data, test_input_data
    return train_data, test_data


def test_kernel_ensemble():
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=0.1,
                      n_jobs=1,
                      industrial_strategy='kernel_automl',
                      industrial_strategy_params={},
                      logging_level=60)
    industrial = FedotIndustrial(**api_config)
    train_data, test_data = classification_data()
    industrial.fit(train_data)
    predict = industrial.predict(test_data)

    assert predict is not None

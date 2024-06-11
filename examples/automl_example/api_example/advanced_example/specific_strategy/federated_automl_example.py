from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator

api_config = dict(problem='classification',
                  metric='f1',
                  timeout=0.1,
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

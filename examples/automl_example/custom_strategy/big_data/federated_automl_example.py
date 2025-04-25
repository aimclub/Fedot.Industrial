from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_AUTOML_LEARNING_CONFIG, \
    DEFAULT_CLF_AUTOML_CONFIG, DEFAULT_CLF_API_CONFIG
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def run_federated_automl_example(timeout: int = 10):
    industrial_config = {'problem': 'classification',
                         'data_type': 'time_series',
                         'learning_strategy': 'federated_automl'}

    learning_config = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': {**DEFAULT_AUTOML_LEARNING_CONFIG, 'timeout': timeout},
                       'optimisation_loss': {'quality_loss': 'f1'}}

    api_config = {'industrial_config': industrial_config,
                  'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
                  'learning_config': learning_config,
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    train_data, test_data = TimeSeriesDatasetsGenerator(num_samples=1800,
                                                        task='classification',
                                                        max_ts_len=50,
                                                        binary=True,
                                                        test_size=0.5,
                                                        multivariate=False).generate_data()
    dataset_dict = dict(train_data=train_data, test_data=test_data)
    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=('f1', 'accuracy')).eval(dataset=dataset_dict,
                                                                   finetune=False)
    return result_dict


if __name__ == '__main__':
    result = run_federated_automl_example(timeout=2)
    print(result)

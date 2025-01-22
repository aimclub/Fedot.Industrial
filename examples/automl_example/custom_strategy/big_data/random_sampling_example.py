from examples.automl_example.custom_strategy.big_data.big_dataset_utils import create_big_dataset
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG

cur_params = {'rank': None}
sampling_algorithm = {'CUR': cur_params}


def eval_fedot_on_fold(dataset_name, fold):
    return create_big_dataset(dataset_name, fold)


INDUSTRIAL_PARAMS = {'data_type': 'tensor',
                     'learning_strategy': 'big_dataset',
                     'sampling_strategy': sampling_algorithm
                     }

AUTOML_LEARNING_STRATEGY = DEFAULT_AUTOML_LEARNING_CONFIG
COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'classification',
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'f1'}}
INDUSTRIAL_CONFIG = {'problem': 'classification',
                     'strategy': 'tabular',
                     'strategy_params': INDUSTRIAL_PARAMS
                     }
API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    metric_by_fold = {}
    finetune = False
    metric_names = ('f1', 'accuracy')
    dataset_name = 'airlines'
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=20,
                      pop_size=3,
                      early_stopping_iterations=10,
                      early_stopping_timeout=30,
                      optimizer_params={'mutation_agent': 'bandit',
                                        'mutation_strategy': 'growth_mutation_strategy'},
                      with_tunig=False,
                      preset='classification_tabular',
                      industrial_strategy_params={'data_type': 'tensor',
                                                  'learning_strategy': 'big_dataset',
                                                  'sampling_strategy': sampling_algorithm
                                                  },
                      n_jobs=-1,
                      logging_level=20)
    for fold in range(10):
        dataset_dict = eval_fedot_on_fold(dataset_name, fold)
        result_dict = ApiTemplate(api_config=API_CONFIG,
                                  metric_list=metric_names).eval(dataset=dataset_dict,
                                                                 finetune=finetune)
        metric_by_fold.update({fold: result_dict})
    _ = 1

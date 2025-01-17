from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH, M4_FORECASTING_LENGTH

DEEPAR_LEARNING_PARAMS = {'epochs': 150,
                          'lr': 0.001,
                          'device': 'cpu'
                          }

model_to_compare = [{0: [('deepar_model', DEEPAR_LEARNING_PARAMS)]},
                    {},
                    {0: ['ar']}
                    ]
model_name = ['deepar', 'industrial', 'ar']
finutune_existed_model = [True, False, True]

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'ts_forecasting',
                 'task_params': {'forecast_length': 14},
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
AUTOML_LEARNING_STRATEGY = dict(timeout=3,
                                n_jobs=2,
                                logging_level=40)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'rmse'}}

INDUSTRIAL_CONFIG = {'problem': 'ts_forecasting',
                     'task_params': {'forecast_length': 14}}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    result_dict = {}
    for dataset in M4_FORECASTING_BENCH:
        prefix = dataset[0]
        horizon = M4_FORECASTING_LENGTH[prefix]
        dataset_dict = {}
        dataset_name = {'benchmark': 'M4',
                        'dataset': dataset,
                        'task_params': {'forecast_length': horizon}}
        API_CONFIG['industrial_config']['task_params']['forecast_length'] = horizon
        API_CONFIG['automl_config']['task_params']['forecast_length'] = horizon
        for model, name, finetune_strategy in zip(model_to_compare, model_name, finutune_existed_model):
            result_dict = ApiTemplate(
                api_config=API_CONFIG,
                metric_list=('rmse', 'mae')).eval(
                dataset=dataset_name,
                initial_assumption=model,
                finetune=finetune_strategy)
            dataset_dict.update({name: {'metric': result_dict['metrics'],
                                        'forecast': result_dict['labels']}})
        result_dict.update({dataset: dataset_dict})
    _ = 1

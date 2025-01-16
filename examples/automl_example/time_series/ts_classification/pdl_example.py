from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG

DATASET_NAME = 'Lightning7'
COMPARISON_DICT = dict(pairwise_approach=['quantile_extractor', 'pdl_clf'],
                       baseline=['quantile_extractor', 'rf'])
METRIC_NAMES = ('f1', 'accuracy', 'precision', 'roc_auc')

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'classification',
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
AUTOML_LEARNING_STRATEGY = dict(timeout=5,
                                pop_size=10,
                                n_jobs=4)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'f1'}}

INDUSTRIAL_CONFIG = {'problem': 'classification'}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    for approach, node_list in COMPARISON_DICT.items():
        result_dict = ApiTemplate(api_config=API_CONFIG,
                                  metric_list=METRIC_NAMES).eval(dataset=DATASET_NAME,
                                                                 initial_assumption=node_list,
                                                                 finetune=True)
        print(f'Approach: {approach}. Metrics: {result_dict["metrics"]}')

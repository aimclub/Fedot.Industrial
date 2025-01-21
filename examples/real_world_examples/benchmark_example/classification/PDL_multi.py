from benchmark.benchmark_TSC import BenchmarkTSC
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH

COMPARISON_DICT = dict(pairwise_approach={0: ['quantile_extractor', 'pdl_clf']},
                       baseline={0: ['quantile_extractor', 'rf']})
METRIC_NAMES = ('f1', 'accuracy', 'precision', 'roc_auc')

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
COMPUTE_CONFIG['output_folder'] = PROJECT_PATH + '/benchmark/results/'
AUTOML_CONFIG = {'task': 'classification',
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
AUTOML_LEARNING_STRATEGY = dict(timeout=2,
                                pop_size=10,
                                n_jobs=-1,
                                num_of_generations=15)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'accuracy'}}

INDUSTRIAL_CONFIG = {'problem': 'classification'}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    BenchmarkTSC(experiment_setup=API_CONFIG,
                 use_small_datasets=False,
                 metric_names=METRIC_NAMES,
                 initial_assumptions=COMPARISON_DICT,
                 finetune=False).run()
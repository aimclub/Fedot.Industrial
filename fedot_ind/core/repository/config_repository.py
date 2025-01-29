from enum import Enum
from os import cpu_count


class ComputeConfigConstant(Enum):
    DEFAULT_COMPUTE_CONFIG = {'backend': 'cpu',
                              'distributed': dict(processes=False,
                                                  n_workers=2,
                                                  threads_per_worker=2,
                                                  memory_limit=0.3
                                                  ),
                              'output_folder': './results',
                              'use_cache': None,
                              'automl_folder': {'optimisation_history': './results/opt_hist',
                                                'composition_results': './results/comp_res'}}


class AutomlLearningConfigConstant(Enum):
    DEFAULT_AUTOML_CONFIG = dict(timeout=10,
                                 pop_size=5,
                                 early_stopping_iterations=10,
                                 early_stopping_timeout=10,
                                 with_tuning=False,
                                 n_jobs=-1)


class AutomlConfigConstant(Enum):
    DEFAULT_SUBCONFIG = {'use_automl': True,
                         'optimisation_strategy': {'optimisation_strategy':
                                                   {'mutation_agent': 'random',
                                                    'mutation_strategy': 'growth_mutation_strategy'},
                                                   'optimisation_agent': 'Industrial'}}
    DEFAULT_CLF_AUTOML_CONFIG = {'task': 'classification', **DEFAULT_SUBCONFIG}
    DEFAULT_REG_AUTOML_CONFIG = {'task': 'regression', **DEFAULT_SUBCONFIG}
    DEFAULT_TSF_AUTOML_CONFIG = {'task': 'ts_forecasting', **DEFAULT_SUBCONFIG}


class LearningConfigConstant(Enum):
    DEFAULT_SUBCONFIG = {'learning_strategy': 'from_scratch',
                         'learning_strategy_params': AutomlLearningConfigConstant.DEFAULT_AUTOML_CONFIG.value}
    DEFAULT_CLF_LEARNING_CONFIG = {'optimisation_loss': {'quality_loss': 'accuracy'}, **DEFAULT_SUBCONFIG}
    DEFAULT_REG_LEARNING_CONFIG = {'optimisation_loss': {'quality_loss': 'rmse'}, **DEFAULT_SUBCONFIG}


DEFAULT_AUTOML_LEARNING_CONFIG = AutomlLearningConfigConstant.DEFAULT_AUTOML_CONFIG.value
DEFAULT_COMPUTE_CONFIG = ComputeConfigConstant.DEFAULT_COMPUTE_CONFIG.value
DEFAULT_CLF_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_CLF_AUTOML_CONFIG.value
DEFAULT_REG_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_REG_AUTOML_CONFIG.value
DEFAULT_TSF_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_TSF_AUTOML_CONFIG.value

DEFAULT_CLF_LEARNING_CONFIG = LearningConfigConstant.DEFAULT_CLF_LEARNING_CONFIG.value
DEFAULT_REG_LEARNING_CONFIG = LearningConfigConstant.DEFAULT_REG_LEARNING_CONFIG.value
from enum import Enum
from os import cpu_count


class ComputeConfigConstant(Enum):
    DEFAULT_COMPUTE_CONFIG = {'backend': 'cpu',
                              'distributed': dict(processes=False,
                                                  n_workers=1,
                                                  threads_per_worker=round(cpu_count() / 2),
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
                                 n_jobs=1)


class AutomlConfigConstant(Enum):
    DEFAULT_SUBCONFIG = {'use_automl': True,
                         'optimisation_strategy': {'optimisation_strategy':
                                                   {'mutation_agent': 'random',
                                                    'mutation_strategy': 'growth_mutation_strategy'},
                                                   'optimisation_agent': 'Industrial'}}
    DEFAULT_CLF_AUTOML_CONFIG = {'task': 'classification', **DEFAULT_SUBCONFIG}
    DEFAULT_REG_AUTOML_CONFIG = {'task': 'regression', **DEFAULT_SUBCONFIG}
    DEFAULT_TSF_AUTOML_CONFIG = {'task': 'ts_forecasting', **DEFAULT_SUBCONFIG}


DEFAULT_AUTOML_LEARNING_CONFIG = AutomlLearningConfigConstant.DEFAULT_AUTOML_CONFIG.value
DEFAULT_COMPUTE_CONFIG = ComputeConfigConstant.DEFAULT_COMPUTE_CONFIG.value
DEFAULT_CLF_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_CLF_AUTOML_CONFIG.value
DEFAULT_REG_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_REG_AUTOML_CONFIG.value
DEFAULT_TSF_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_TSF_AUTOML_CONFIG.value

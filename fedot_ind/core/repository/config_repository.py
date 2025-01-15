from enum import Enum
from os import cpu_count


class ApiConfigConstant(Enum):
    DEFAULT_AUTOML_CONFIG = dict(timeout=10,
                                 pop_size=5,
                                 early_stopping_iterations=10,
                                 early_stopping_timeout=10,
                                 with_tuning=False
                                 )
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


DEFAULT_AUTOML_LEARNING_CONFIG = ApiConfigConstant.DEFAULT_AUTOML_CONFIG.value
DEFAULT_COMPUTE_CONFIG = ApiConfigConstant.DEFAULT_COMPUTE_CONFIG.value

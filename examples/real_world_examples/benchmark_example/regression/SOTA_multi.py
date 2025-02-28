import gc

from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_REG_API_CONFIG
from fedot_ind.core.repository.constanst_repository import MULTI_REG_BENCH


def get_benchmark_setup():
    RESNET_LEARNING_PARAMS = {'epochs': 150,
                              'lr': 0.001,
                              'device': 'cpu'}
    bagging_ensemble = {0: ['industrial_freq_reg'],
                        1: ['industrial_stat_reg'],
                        'head': ['bagging']}

    model_to_compare = [bagging_ensemble,
                        {},
                        {0: [('resnet_model', RESNET_LEARNING_PARAMS)]}]
    model_name = ['stat_reg', 'industrial', 'resnet']
    finutune_existed_model = [True, False, True]
    BENCHMARK = 'UCR_REG'
    BENCHMARK_PARAMS = {'datasets': MULTI_REG_BENCH,
                        'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
    return BENCHMARK, BENCHMARK_PARAMS


if __name__ == "__main__":
    gc.collect()
    BENCHMARK_CONFIG = {'task': 'regression',
                        'timeout': 3,
                        'n_workers': 2,
                        'threads_per_worker': 4,
                        'with_tuning': False,
                        'logging_level': 20}
    UPD_API_CONFIG = ApiConfigCheck().update_config_with_kwargs(DEFAULT_REG_API_CONFIG, **BENCHMARK_CONFIG)
    api_agent = ApiTemplate(api_config=UPD_API_CONFIG, metric_list=('rmse', 'mae'))
    BENCHMARK, BENCHMARK_PARAMS = get_benchmark_setup()
    api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                 benchmark_params=BENCHMARK_PARAMS)

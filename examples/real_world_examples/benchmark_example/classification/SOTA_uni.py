import gc

from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_API_CONFIG
from fedot_ind.core.repository.constanst_repository import UNI_CLF_BENCH

gc.collect()


def get_benchmark_setup():
    RESNET_LEARNING_PARAMS = {'epochs': 150,
                              'lr': 0.001,
                              'device': 'cpu'
                              }

    model_to_compare = [{0: ['industrial_freq_clf'],
                         1: ['industrial_stat_clf'],
                         'head': ['bagging']},
                        {},
                        {0: [('resnet_model', RESNET_LEARNING_PARAMS)]}]
    model_name = ['stat_clf', 'industrial', 'resnet']
    finetune_existed_model = [True, False, True]
    BENCHMARK = 'UCR_CLF'
    BENCHMARK_PARAMS = {'datasets': UNI_CLF_BENCH,
                        'model_to_compare': (model_to_compare, model_name, finetune_existed_model)}
    return BENCHMARK, BENCHMARK_PARAMS


if __name__ == "__main__":
    BENCHMARK_CONFIG = {'task': 'classification',
                        'timeout': 3,
                        'n_workers': 2,
                        'threads_per_worker': 4,
                        'with_tuning': False,
                        'logging_level': 20}

    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG, **BENCHMARK_CONFIG)

    api_agent = ApiTemplate(api_config=api_config, metric_list=('accuracy', 'f1'))
    BENCHMARK, BENCHMARK_PARAMS = get_benchmark_setup()
    api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                 benchmark_params=BENCHMARK_PARAMS)

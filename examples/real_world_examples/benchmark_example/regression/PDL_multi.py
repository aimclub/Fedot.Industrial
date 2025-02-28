from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_REG_API_CONFIG
from fedot_ind.core.repository.constanst_repository import MULTI_REG_BENCH


def get_pdl_model_to_compare():
    model_to_compare = [{0: ['quantile_extractor', 'treg']},
                        {0: ['quantile_extractor', 'pdl_reg']}]
    model_name = ['treg', 'pdl_reg']
    finetune_existed_model = [True, True]
    return model_to_compare, model_name, finetune_existed_model


if __name__ == "__main__":
    BENCHMARK_CONFIG = {'task': 'regression',
                        'timeout': 3,
                        'n_workers': 2,
                        'threads_per_worker': 4,
                        'with_tuning': False,
                        'logging_level': 20}
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_REG_API_CONFIG, **BENCHMARK_CONFIG)
    api_agent = ApiTemplate(api_config=api_config, metric_list=('rmse', 'mae'))
    api_agent.evaluate_benchmark(benchmark_name='UCR_MULTI',
                                 benchmark_params={'metadata': None,
                                                   'datasets': MULTI_REG_BENCH,
                                                   'model_to_compare': get_pdl_model_to_compare()})

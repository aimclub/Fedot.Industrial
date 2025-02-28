from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_TSF_API_CONFIG
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH, M4_FORECASTING_LENGTH


def get_m4_model_to_compare():
    model_to_compare = [{0: ['lagged_forecaster']},
                        {},
                        {0: [('deepar_model', {'epochs': 150, 'lr': 0.001, 'device': 'cpu'})]},
                        {0: ['ar']}]
    model_name = ['lagged_regression', 'industrial', 'deepar', 'ar']
    finetune_existed_model = [True, False, True, True]
    return model_to_compare, model_name, finetune_existed_model


if __name__ == "__main__":
    BENCHMARK_CONFIG = {'task': 'ts_forecasting',
                        'task_params': {'forecast_length': 14},
                        'timeout': 5,
                        'n_workers': 2,
                        'threads_per_worker': 2,
                        'with_tuning': False,
                        'logging_level': 20}

    UPD_API_CONFIG = ApiConfigCheck().update_config_with_kwargs(DEFAULT_TSF_API_CONFIG, **BENCHMARK_CONFIG)
    api_agent = ApiTemplate(api_config=UPD_API_CONFIG, metric_list=('rmse', 'mae'))
    EVALUATED = []
    DATASETS = [x for x in M4_FORECASTING_BENCH if x not in EVALUATED]
    api_agent.evaluate_benchmark(benchmark_name='M4',
                                 benchmark_params={'metadata': M4_FORECASTING_LENGTH,
                                                   'datasets': DATASETS,
                                                   'model_to_compare': get_m4_model_to_compare()})

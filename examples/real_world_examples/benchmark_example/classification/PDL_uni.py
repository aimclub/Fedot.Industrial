import pandas as pd

from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_API_CONFIG
from fedot_ind.core.repository.constanst_repository import UNI_CLF_BENCH
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH


def filter_datasets(UNI_CLF_BENCH, max_classes: int = 10, max_samples: int = 1000):
    UNI_CLF_BENCH_METADATA = pd.read_csv(PROJECT_PATH + '/fedot_ind/core/repository/data/ts_benchmark_metadata.csv')
    filtered_by_classes = UNI_CLF_BENCH_METADATA[UNI_CLF_BENCH_METADATA['Class'] <= max_classes]
    filtered_by_samples = filtered_by_classes[filtered_by_classes['Train '] <= max_samples]
    filtered_by_samples = filtered_by_samples[filtered_by_samples['Test '] <= max_samples]['Name'].values.tolist()
    UNI_CLF_BENCH = [x for x in UNI_CLF_BENCH if x in filtered_by_samples]
    UNI_CLF_BENCH_METADATA = UNI_CLF_BENCH_METADATA[UNI_CLF_BENCH_METADATA['Name'].isin(filtered_by_samples)]
    return UNI_CLF_BENCH, UNI_CLF_BENCH_METADATA


def get_pdl_model_to_compare():
    model_to_compare = [{0: ['quantile_extractor', 'rf']},
                        {0: ['quantile_extractor', 'pdl_clf']}]
    model_name = ['rf', 'pdl_clf']
    finetune_existed_model = [True, True]
    return model_to_compare, model_name, finetune_existed_model


if __name__ == "__main__":
    UNI_CLF_BENCH, UNI_CLF_BENCH_METADATA = filter_datasets(UNI_CLF_BENCH)
    BENCHMARK_CONFIG = {'task': 'classification',
                        'timeout': 3,
                        'pop_size': 10,
                        'n_jobs': -1,
                        'num_of_generations': 15}
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG, **BENCHMARK_CONFIG)
    api_agent = ApiTemplate(api_config=api_config, metric_list=('f1', 'accuracy', 'precision', 'roc_auc'))
    api_agent.evaluate_benchmark(benchmark_name='UCR_CLF',
                                 benchmark_params={'benchmark_folder': '.',
                                                   'metadata': None,
                                                   'datasets': UNI_CLF_BENCH,
                                                   'model_to_compare': get_pdl_model_to_compare()})

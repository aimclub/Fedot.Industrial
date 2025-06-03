from examples.automl_example.custom_strategy.big_data.big_dataset_utils import create_big_dataset
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_AUTOML_LEARNING_CONFIG, \
    DEFAULT_CLF_AUTOML_CONFIG


def run_random_sampling_example(timeout: int = 10, fold_amount: int = 10) -> dict:
    industrial_params = {'data_type': 'tensor',
                         'learning_strategy': 'big_dataset',
                         'sampling_strategy': {'CUR': {'rank': None}}}

    learning_config = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': {**DEFAULT_AUTOML_LEARNING_CONFIG, 'timeout': timeout},
                       'optimisation_loss': {'quality_loss': 'f1'}}

    industrial_config = {'problem': 'classification',
                         'strategy': 'tabular',
                         'strategy_params': industrial_params}

    api_config = {'industrial_config': industrial_config,
                  'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
                  'learning_config': learning_config,
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    metric_by_fold = {}
    for fold in range(fold_amount):
        dataset_dict = create_big_dataset('airlines', fold)
        result_dict = ApiTemplate(api_config=api_config,
                                  metric_list=('f1', 'accuracy', 'precision', 'roc_auc')).eval(dataset=dataset_dict,
                                                                                               finetune=False)
        metric_by_fold.update({fold: result_dict})
    return metric_by_fold


if __name__ == "__main__":
    metric_by_fold = run_random_sampling_example(timeout=2)
    print(metric_by_fold)

from fedot_ind.tools.example_utils import create_feature_generator_strategy
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG, DEFAULT_CLF_AUTOML_CONFIG


def run_multimodal_example(timeout: int = 10):
    feature_generator, sampling_dict = create_feature_generator_strategy()

    industrial_params = {'feature_generator': feature_generator,
                         'data_type': 'tensor',
                         'learning_strategy': 'ts2tabular',
                         'sampling_strategy': sampling_dict}

    dataset_name = 'Lightning7'
    metric_names = ('f1', 'accuracy')

    learning_config = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': {**DEFAULT_AUTOML_LEARNING_CONFIG, 'timeout': timeout},
                       'optimisation_loss': {'quality_loss': 'f1'}}
    industrial_config = {'problem': 'classification',
                         'strategy': 'tabular',
                         'strategy_params': industrial_params
                         }
    api_config = {'industrial_config': industrial_config,
                  'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
                  'learning_config': learning_config,
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    multimodal_pipeline = {0: [
        # ('recurrence_extractor', {'window_size': 30, 'stride': 5, 'image_mode': True}),
        ('quantile_extractor', {'window_size': 30, 'stride': 5, 'image_mode': True}),
        ('resnet_model', {'epochs': 1, 'batch_size': 16, 'model_name': 'ResNet50'})
    ]}

    explain_config = {'method': 'recurrence',
                      'samples': 1,
                      'metric': 'mean'}

    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=metric_names).eval(dataset=dataset_name,
                                                             finetune=False,
                                                             initial_assumption=multimodal_pipeline)
    result_dict['industrial_model'].explain(explain_config)


if __name__ == "__main__":
    run_multimodal_example(timeout=5)

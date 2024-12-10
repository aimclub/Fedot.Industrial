from examples.automl_example.custom_strategy.big_data.big_dataset_utils import create_big_dataset
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

cur_params = {'rank': None}
sampling_algorithm = {'CUR': cur_params}

def eval_fedot_on_fold(dataset_name,fold):
    return create_big_dataset(dataset_name,fold)

if __name__ == "__main__":
    metric_by_fold = {}
    finetune = False
    metric_names = ('f1', 'accuracy')
    dataset_name = 'airlines'
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=20,
                      pop_size=3,
                      early_stopping_iterations=10,
                      early_stopping_timeout=30,
                      optimizer_params={'mutation_agent': 'bandit',
                                        'mutation_strategy': 'growth_mutation_strategy'},
                      with_tunig=False,
                      preset='classification_tabular',
                      industrial_strategy_params={'data_type': 'tensor',
                                                  'learning_strategy': 'big_dataset',
                                                  'sampling_strategy': sampling_algorithm
                                                  },
                      n_jobs=-1,
                      logging_level=20)
    for fold in range(10):
        dataset_dict = create_big_dataset(dataset_name, fold)
        result_dict = ApiTemplate(api_config=api_config,
                                  metric_list=metric_names).eval(dataset=dataset_dict,
                                                                 finetune=finetune)
        metric_by_fold.update({fold:result_dict})
    _ = 1

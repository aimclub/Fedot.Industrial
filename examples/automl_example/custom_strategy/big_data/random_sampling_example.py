from examples.automl_example.custom_strategy.big_data.big_dataset_utils import create_big_dataset
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

cur_params = {'rank': None}
sampling_algorithm = {'CUR': cur_params}

if __name__ == "__main__":
    dataset_dict = create_big_dataset()
    finetune = False
    metric_names = ('f1', 'accuracy')
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=40,
                      pop_size=10,
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

    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=metric_names).eval(dataset=dataset_dict,
                                                             finetune=finetune)
    metrics = result_dict['metrics']
    metrics.to_csv('./metrics.csv')
    hist = result_dict['industrial_model'].save_optimization_history(return_history=True)
    result_dict['industrial_model'].vis_optimisation_history(hist)
    result_dict['industrial_model'].save_best_model()
    _ = 1

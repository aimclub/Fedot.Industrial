from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = 'Lightning7'
    finetune = True
    metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=5,
                      pop_size=10,
                      with_tunig=False,
                      n_jobs=2,
                      logging_level=20)
    init_assumption_pdl = ['quantile_extractor', 'pdl_clf']
    init_assumption_rf = ['quantile_extractor', 'rf']
    comparasion_dict = dict(pairwise_approach=init_assumption_pdl,
                            baseline=init_assumption_rf)
    for approach in comparasion_dict.keys():
        result_dict = ApiTemplate(api_config=api_config,
                                  metric_list=metric_names).eval(dataset=dataset_name,
                                                                 initial_assumption=comparasion_dict[approach],
                                                                 finetune=finetune)
        metrics = result_dict['metrics']
        print(f'Approach - {approach}. Metrics - {metrics}')
    _ = 1

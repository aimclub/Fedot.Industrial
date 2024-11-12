from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = 'AppliancesEnergy'  # BeijingPM10Quality
    finetune = False
    api_config = dict(problem='regression',
                      metric='rmse',
                      timeout=0.1,
                      n_jobs=2,
                      logging_level=20)
    metric_names = ('r2', 'rmse', 'mae')
    init_assumption_pdl = ['quantile_extractor', 'pdl_reg']
    init_assumption_rf = ['quantile_extractor', 'treg']
    comparasion_dict = dict(pairwise_approach=init_assumption_pdl,
                            baseline=init_assumption_rf)
    for approach in comparasion_dict.keys():
        result_dict = ApiTemplate(api_config=api_config,
                                  metric_list=metric_names).eval(dataset=dataset_name,
                                                                 initial_assumption=comparasion_dict[approach],
                                                                 finetune=finetune)
        metrics = result_dict['metrics']
        print(f'Approach - {approach}. Metrics - {metrics}')

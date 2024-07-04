from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = 'Lightning7'
    finetune = False
    metric_names = ('f1', 'accuracy')
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=5,
                      pop_size=5,
                      with_tuning=False,
                      cv_folds=3,
                      n_jobs=-1,
                      logging_level=10)

    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=('f1', 'accuracy')).eval(dataset=dataset_name,
                                                                   finetune=finetune)

    opt_hist = result_dict['industrial_model'].save_optimization_history(return_history=True)
    opt_hist = result_dict['industrial_model'].vis_optimisation_history(
        opt_history_path=opt_hist, return_history=True)

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = 'Handwriting'
    finetune = False
    metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=1,
                      pop_size=10,
                      with_tunig=False,
                      n_jobs=2,
                      logging_level=20)
    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=metric_names).eval(dataset=dataset_name, finetune=finetune)
    print(result_dict['metrics'])

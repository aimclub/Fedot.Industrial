from our_approach.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = 'Datasetname'
    finetune = False
    metric_names = ('f1')
    api_config = dict(problem='regression',
                      metric='f1',
                      timeout=1,
                      pop_size=10,
                      with_tunig=False,
                      n_jobs=2,
                      logging_level=20)
    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=metric_names).eval(dataset=dataset_name, finetune=finetune)
    print(result_dict['metrics'])

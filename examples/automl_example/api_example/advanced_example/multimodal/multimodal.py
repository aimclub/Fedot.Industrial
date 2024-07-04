from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = 'Earthquakes'
    finetune = False
    metric_names = ('f1', 'accuracy')
    multimodal_pipeline = {'recurrence_extractor': {
        'window_size': 30,
        'stride': 5,
        'image_mode': True},
        'resnet_model': {
            'epochs': 30,
            'batch_size': 16,
            'model_name': 'ResNet50'}}

    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=0.1,
                      pop_size=5,
                      with_tuning=False,
                      cv_folds=3,
                      n_jobs=-1,
                      logging_level=10)

    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=('f1', 'accuracy')).eval(dataset=dataset_name,
                                                                   finetune=finetune,
                                                                   initial_assumption=multimodal_pipeline)
    _ = 1

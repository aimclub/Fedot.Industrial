from fedot_ind.tools.example_utils import industrial_common_modelling_loop

if __name__ == "__main__":
    dataset_name = dict(benchmark='valve1',
                        dataset='1')
    prediction_window = 10
    finetune = False
    metric_names = ('nab')
    metric_names = ('accuracy')
    api_config = dict(problem='classification',
                      metric='accuracy',
                      timeout=1,
                      pop_size=10,
                      industrial_task_params={'industrial_task': 'anomaly_detection',
                                              'detection_window': prediction_window,
                                              'data_type': 'time_series'},
                      with_tunig=False,
                      n_jobs=2,
                      logging_level=20)

    model, labels, metrics = industrial_common_modelling_loop(
        api_config=api_config, dataset_name=dataset_name, finetune=finetune, metric_names=metric_names)
    print(metrics)

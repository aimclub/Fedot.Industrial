from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = dict(benchmark='valve1',
                        dataset='1')
    prediction_window = 10
    finetune = False
    metric_names = ('nab', 'accuracy')
    api_config = dict(
        problem='classification',
        metric='accuracy',
        timeout=1,
        pop_size=10,
        industrial_strategy='anomaly_detection',
        industrial_task_params={
            'detection_window': prediction_window,
            'data_type': 'time_series'},
        with_tuning=False,
        n_jobs=2,
        logging_level=20)
    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=metric_names).eval(dataset=dataset_name, finetune=finetune)
    print(result_dict['metrics'])

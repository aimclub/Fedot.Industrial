from fedot_ind.tools.example_utils import industrial_common_modelling_loop

if __name__ == "__main__":
    dataset_name = dict(benchmark='valve1',
                        dataset='1')
    finetune = False
    metric_names = ('nab')
    api_config = dict(problem='anomaly_detection',
                      metric='f1',
                      timeout=2,
                      pop_size=10,
                      with_tunig=False,
                      n_jobs=2,
                      logging_level=20)

    model, labels, metrics = industrial_common_modelling_loop(
        api_config=api_config, dataset_name=dataset_name, finetune=finetune)
    print(metrics)

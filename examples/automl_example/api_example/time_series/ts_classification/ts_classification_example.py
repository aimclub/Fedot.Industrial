from fedot_ind.tools.example_utils import industrial_common_modelling_loop

if __name__ == "__main__":
    dataset_name = 'Handwriting'
    finetune = False
    metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=2,
                      pop_size=10,
                      with_tunig=False,
                      n_jobs=2,
                      logging_level=20)

    model, labels, metrics = industrial_common_modelling_loop(
        api_config=api_config, dataset_name=dataset_name, finetune=finetune)
    print(metrics)

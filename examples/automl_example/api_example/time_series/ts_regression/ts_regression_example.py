from fedot_ind.tools.example_utils import industrial_common_modelling_loop

if __name__ == "__main__":
    dataset_name = 'ApplianceEnergy'  # BeijingPM10Quality
    finetune = False
    api_config = dict(problem='regression',
                      metric='rmse',
                      timeout=5,
                      n_jobs=2,
                      logging_level=20)
    metric_names = ('r2', 'rmse', 'mae')
    model, labels, metrics = industrial_common_modelling_loop(
        api_config=api_config, dataset_name=dataset_name, finetune=finetune)
    print(metrics)

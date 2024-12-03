from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

if __name__ == "__main__":
    dataset_name = 'AppliancesEnergy'  # BeijingPM10Quality
    finetune = False
    api_config = dict(problem='regression',
                      metric='rmse',
                      timeout=1,
                      n_jobs=2,
                      logging_level=20)
    metric_names = ('r2', 'rmse', 'mae')
    result_dict = ApiTemplate(api_config=api_config,
                              metric_list=('f1', 'accuracy')).eval(dataset=dataset_name, finetune=finetune)
    print(result_dict['metrics'])

from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

finetune = False
metric_names = ('rmse', 'mae')
if __name__ == "__main__":
    dataset_name = {'benchmark': 'M4',
                    'dataset': 'D3257',
                    'task_params': {'forecast_length': 14}}
    initial_assumptions = {
        'nbeats': PipelineBuilder().add_node('nbeats_model'),
        'industiral': PipelineBuilder().add_node(
            'eigen_basis',
            params={
                'low_rank_approximation': False,
                'rank_regularization': 'explained_dispersion'}).add_node('ar')
    }
    for assumption in initial_assumptions.keys():
        api_config = dict(problem='ts_forecasting',
                          metric='rmse',
                          timeout=5,
                          task_params={'forecast_length': 14},
                          n_jobs=2,
                          initial_assumption=initial_assumptions[assumption],
                          logging_level=20)
        result_dict = ApiTemplate(api_config=api_config,
                                  metric_list=('f1', 'accuracy')).eval(dataset=dataset_name, finetune=finetune)

        current_metric = result_dict['metrics']
        print(f'{assumption} have metrics - {current_metric}')

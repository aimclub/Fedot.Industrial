from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.tools.example_utils import industrial_common_modelling_loop

if __name__ == "__main__":
    dataset_name = {'benchmark': 'M4',
                    'dataset': 'D3257'}
    finetune = False
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
        metric_names = ('r2', 'rmse', 'mae')
        model, labels, metrics = industrial_common_modelling_loop(
            api_config=api_config, dataset_name=dataset_name, finetune=finetune)
        finetune = False
        print(f'{assumption} have metrics - {metrics}')

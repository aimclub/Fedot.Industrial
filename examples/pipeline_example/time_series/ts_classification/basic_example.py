from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from examples.example_utils import evaluate_metric
from examples.example_utils import init_input_data
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

model_dict = {'basic_quantile': PipelineBuilder().add_node('quantile_extractor',
                                                           params={'window_size': 10,
                                                                   'stride': 5}).add_node('rf'),
              'basic_topological': PipelineBuilder().add_node('topological_extractor',
                                                              params={'window_size': 10,
                                                                      'stride': 5}).add_node('rf'),
              'basic_recurrence': PipelineBuilder().add_node('recurrence_extractor',
                                                             params={'window_size': 10,
                                                                     'stride': 5}).add_node('rf'),
              'advanced_quantile': PipelineBuilder().add_node('fourier_basis').add_node('quantile_extractor',
                                                                                        params={
                                                                                            'window_size': 10}).add_node(
                  'rf'),
              'advanced_topological': PipelineBuilder().add_node('eigen_basis').add_node('topological_extractor',
                                                                                         params={'stride': 20,
                                                                                                 'window_size': 10}).add_node(
                  'rf'),
              'advanced_reccurence': PipelineBuilder().add_node('wavelet_basis').add_node(
                  'recurrence_extractor',
                  params={'window_size': 0}).add_node(
                  'rf')
              }
metric_dict = {}
train_data, test_data = DataLoader(dataset_name='Ham').load_data()

if __name__ == "__main__":
    with IndustrialModels():
        for model in model_dict.keys():
            pipeline = model_dict[model].build()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])
            pipeline.fit(input_data)
            features = pipeline.predict(val_data).predict
            metric = evaluate_metric(target=test_data[1], prediction=features)
            metric_dict.update({model: metric})
        print(metric_dict)

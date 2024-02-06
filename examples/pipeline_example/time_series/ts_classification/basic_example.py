import matplotlib
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

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

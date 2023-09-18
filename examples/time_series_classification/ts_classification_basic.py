import numpy as np
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from examples.fedot.fedot_ex import init_input_data
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from sklearn.metrics import f1_score, roc_auc_score

model_dict = {'basic_quantile': PipelineBuilder().add_node('quantile_extractor',
                                                           params={'window_size': 10,
                                                                   'window_mode': True}).add_node('rf'),
              'basic_topological': PipelineBuilder().add_node('topological_extractor',
                                                              params={'window_size': 10}).add_node('rf'),
              'basic_recurrence': PipelineBuilder ().add_node('recurrence_extractor').add_node('rf'),
              'advanced_quantile': PipelineBuilder().add_node('fourier_basis').add_node('quantile_extractor',
                                                                                        params={'window_size': 10,
                                                                                                'window_mode': True}).add_node(
                  'rf'),
              'advanced_topological': PipelineBuilder().add_node('eigen_basis').add_node('topological_extractor',
                                                                                         params={
                                                                                             'window_size': 10}).add_node(
                  'rf'),
              'advanced_reccurence': PipelineBuilder().add_node('wavelet_basis').add_node(
                  'recurrence_extractor').add_node(
                  'rf')
              }
metric_dict = {}
train_data, test_data = DataLoader(dataset_name='Ham').load_data()

with IndustrialModels():
    for model in model_dict.keys():
        pipeline = model_dict[model].build()
        input_data = init_input_data(train_data[0], train_data[1])
        val_data = init_input_data(test_data[0], test_data[1])
        pipeline.fit(input_data)
        features = pipeline.predict(val_data).predict
        if len(np.unique(test_data[1])) > 2:
            metric = f1_score(test_data[1], features, average='weighted')
        else:
            metric = roc_auc_score(test_data[1], features, average='weighted')
        metric_dict.update({model: metric})
    print(metric_dict)
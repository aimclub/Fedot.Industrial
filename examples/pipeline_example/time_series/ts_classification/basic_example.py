import matplotlib
from fedot import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.verification import common_rules

from examples.example_utils import evaluate_metric
from examples.example_utils import init_input_data
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

matplotlib.use('TkAgg')
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
dataset = 'Lightning7'
dataset_multi_dim = 'LSST'
train_data, test_data = DataLoader(dataset_name = dataset_multi_dim).load_data()
ts_clf_operations = [
    'eigen_basis',
    'dimension_reduction',
    'inception_model',
    'rf',
    'minirocket_extractor',
    'normalization',
    'omniscale_model',
    'pca',
    'mlp',
    'quantile_extractor',
    'recurrence_extractor',
    'resample',
    'scaling',
    'signal_extractor',
    'topological_features'
]
# ts_clf_operations = [
#     'eigen_basis',
#     'dimension_reduction',
#     'inception_model',
#     'rf',
#     'minirocket_extractor',
#     'normalization',
#     'omniscale_model',
#     'pca',
#     'mlp',
#     'quantile_extractor',
#     'recurrence_extractor',
#     'resample',
#     'scaling',
#     'signal_extractor',
#     'topological_features'
# ]
if __name__ == "__main__":
    OperationTypesRepository = IndustrialModels().setup_repository()
    #error_pipeline = PipelineBuilder().add_node('scaling').add_node('signal_extractor').add_node('quantile_extractor').add_node('inception_model').build()
    error_pipeline = PipelineBuilder().add_node('scaling').add_node('inception_model').build()
    # error_pipeline = PipelineBuilder().add_node('scaling').add_node('signal_extractor').add_node(
    #     'quantile_extractor').add_node('normalization',branch_idx=1).add_node('signal_extractor',branch_idx=1).add_node(
    #     'quantile_extractor',branch_idx=1).join_branches('logit').build()
    # # error_pipeline = PipelineBuilder().add_node('signal_extractor').add_node(
    # #     'quantile_extractor').add_node('logit').build()
    #     # add_node('inception_model').add_node(
    #     # 'scaling',
    #     # branch_idx=1).join_branches(
    #     # 'logit').build()
    for model in model_dict.keys():
        pipeline = model_dict[model].build()
        input_data = init_input_data(train_data[0], train_data[1])
        val_data = init_input_data(test_data[0], test_data[1])
        model = Fedot(problem='classification',
                      logging_level=20,
                      n_jobs=1,
                      metric='f1',
                      available_operations=ts_clf_operations,
                      timeout=20
                      )
        model.fit(input_data)
        model.current_pipeline.show()
        features = model.predict(val_data)
        metric = evaluate_metric(target=val_data.target, prediction=features)
        metric_dict.update({model: metric})
        model.history.save(f"{dataset}classification_history.json")
        model.history.show.fitness_box(best_fraction=0.5, dpi=100)
        model.history.show.operations_kde(dpi=100)
        model.history.show.operations_animated_bar(save_path=f'./{dataset}_history_animated_bars.gif',
                                                   show_fitness=True, dpi=100)
        print(metric_dict)

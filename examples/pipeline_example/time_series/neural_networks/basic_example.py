import pandas as pd
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.utils.data import init_input_data
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.example_utils import evaluate_metric
from fedot_ind.tools.loader import DataLoader
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.operations.atomized_model import AtomizedModel


def create_ensemble(fitted_model,
                    head_model: str = 'logit'):
    composed_pipeline = Pipeline(PipelineNode(head_model,
                                              nodes_from=[
                                                  PipelineNode(AtomizedModel(
                                                      fitted_model[nested_pipeline]))
                                                  for nested_pipeline in fitted_model]))
    return composed_pipeline


if __name__ == "__main__":
    EPOCHS = 100
    BATCH_SIZE = 32

    dataset_list = [
        'Lightning2',
        'Ham'
    ]

    ensemble_head = [
        'logit',
        'fedot_cls']

    pipeline_dict = {
        # 'inception_model': PipelineBuilder().add_node('inception_model', params={'epochs': EPOCHS,
        #                                                                          'batch_size': BATCH_SIZE}),
        # 'omniscale_model': PipelineBuilder().add_node('omniscale_model', params={'epochs': EPOCHS,
        #                                                                          'batch_size': BATCH_SIZE}),
        # # 'image_model': PipelineBuilder().add_node('recurrence_extractor', params={'image_mode': True}) \
        # #     .add_node('resnet_model', params={'epochs': EPOCHS,
        # #                                       'batch_size': BATCH_SIZE,
        # #                                       'model_name': 'ResNet50'}),
        'rocket_model': PipelineBuilder().add_node('minirocket_extractor', params={'num_features': 20000}).add_node(
            'feature_filter_model', params={'explained_dispersion': 0.9}) \
        .add_node('fedot_cls', params={'timeout': 10}),
        'quantile_rf_model': PipelineBuilder() \
        .add_node('quantile_extractor') \
        .add_node('rf')
    }
    fitted_model = {}
    result_dict = {}

    for dataset in dataset_list:
        train_data, test_data = DataLoader(dataset_name=dataset).load_data()
        input_data = init_input_data(train_data[0], train_data[1])
        val_data = init_input_data(test_data[0], test_data[1])
        metric_dict = {}

        for model in pipeline_dict:
            with IndustrialModels():
                pipeline = pipeline_dict[model].build()
                pipeline.fit(input_data)
                target = pipeline.predict(val_data).predict
                metric = evaluate_metric(
                    target=test_data[1], prediction=target)
            metric_dict.update({model: metric})
            fitted_model.update({model: pipeline})

        with IndustrialModels():
            for head_model in ensemble_head:
                composed_model = create_ensemble(
                    fitted_model, head_model=head_model)
                composed_model.fit(input_data)
                target = composed_model.predict(val_data).predict
                metric = evaluate_metric(
                    target=test_data[1], prediction=target)
                metric_dict.update({f'composed_pipeline_{head_model}': metric})

        result_dict.update({dataset: metric_dict})
    result_df = pd.DataFrame(result_dict)
    print(result_df)

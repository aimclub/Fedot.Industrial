import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from examples.example_utils import evaluate_metric
from fedot_ind.api.utils.data import init_input_data
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":
    EPOCHS = 1
    BATCH_SIZE = 16

    dataset_list = [
        'Lightning7',
        'Ham'
    ]

    pipeline_dict = {
        'recurrence_image_model': PipelineBuilder().add_node('recurrence_extractor', params={'image_mode': True}) \
            .add_node('resnet_model', params={'epochs': EPOCHS,
                                              'batch_size': BATCH_SIZE,
                                              'model_name': 'ResNet18one'}),
        'eigen_xcm_model': PipelineBuilder().add_node(
            'eigen_basis', params={'window_size': 10,
                                   'low_rank_approximation': True}).add_node('xcm_model', params={
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE}),
        'xcm_model': PipelineBuilder().add_node('xcm_model', params={'epochs': EPOCHS,
                                                                     'batch_size': BATCH_SIZE})
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
                pipeline.nodes[1].fitted_operation.explain(val_data)
                pipeline.nodes[0].fitted_operation.model.explain(val_data)
                metric = evaluate_metric(target=test_data[1], prediction=target)
            metric_dict.update({model: metric})
            fitted_model.update({model: pipeline})

        result_dict.update({dataset: metric_dict})
    result_df = pd.DataFrame(result_dict)
    print(result_df)

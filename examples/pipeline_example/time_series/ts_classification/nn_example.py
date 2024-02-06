import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from examples.example_utils import evaluate_metric, init_input_data
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":
    dataset_list = [
                    'Lightning2',
                    'EOGVerticalSignal']
    result_dict = {}
    pipeline_dict = {'inception_model': PipelineBuilder().add_node('inception_model', params={'epochs': 100,
                                                                                              'batch_size': 10}),

                     'quantile_rf_model': PipelineBuilder() \
                         .add_node('quantile_extractor') \
                         .add_node('rf'),
                     'composed_model': PipelineBuilder() \
                         .add_node('inception_model', params={'epochs': 100,
                                                              'batch_size': 10}) \
                         .add_node('quantile_extractor', branch_idx=1) \
                         .add_node('rf', branch_idx=1) \
                         .join_branches('logit')}

    for dataset in dataset_list:
        try:
            train_data, test_data = DataLoader(dataset_name=dataset).load_data()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])
            metric_dict = {}
            for model in pipeline_dict:
                with IndustrialModels():
                    pipeline = pipeline_dict[model].build()
                    pipeline.fit(input_data)
                    target = pipeline.predict(val_data).predict
                    metric = evaluate_metric(target=test_data[1], prediction=target)
                metric_dict.update({model: metric})
            result_dict.update({dataset: metric_dict})
        except Exception:
            print('ERROR')
    result_df = pd.DataFrame(result_dict)
    print(result_df)
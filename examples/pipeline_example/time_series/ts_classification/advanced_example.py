
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from examples.example_utils import evaluate_metric
from fedot_ind.api.utils.data import init_input_data
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

metric_dict = {}

model_dict = {
    'eigen_basis_basic': PipelineBuilder().add_node(
        'eigen_basis',
        params={'window_size': 10, 'low_rank_approximation': False}).add_node(
        'quantile_extractor',
        params={'window_size': 50}).add_node(
        'rf'),

    'eigen_basis_advanced': PipelineBuilder().add_node(
        'eigen_basis', params={'window_size': 10, 'low_rank_approximation': False}).
    add_node('feature_filter_model', params={
        'grouping_level': 0.5}).add_node(
        'quantile_extractor', params={'window_size': 50}).add_node(
        'rf')}

datasets_bad_f1 = [
    'EOGVerticalSignal',
    # 'ScreenType',
    # 'CricketY',
    # 'ElectricDevices',
    'Lightning7'
]

# datasets_good_f1 = [
#     'Car',
#     'ECG5000',
#     "Beef"
# ]
#
# datasets_good_roc = [
#     'Chinatown',
#     'Computers',
#     'Earthquakes',
#     'Ham',
#     'ECG200',
#     'ECGFiveDays',
#     'MiddlePhalanxOutlineCorrect',
#     'MoteStrain',
#     'TwoLeadECG'
# ]

if __name__ == "__main__":
    OperationTypesRepository = IndustrialModels().setup_repository()
    for dataset_name in datasets_bad_f1:
        train_data, test_data = DataLoader(
            dataset_name=dataset_name).load_data()
        input_data = init_input_data(train_data[0], train_data[1])
        val_data = init_input_data(test_data[0], test_data[1])
        for model in model_dict.keys():
            pipeline = model_dict[model].build()
            pipeline.fit(input_data)
            features = pipeline.predict(val_data, 'labels').predict
            metric = evaluate_metric(target=test_data[1], prediction=features)
            metric_dict.update({f'{dataset_name}_{model}': metric})
            print(f'{dataset_name}_{model} - {metric}')
    print(metric_dict)

import os
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from examples.example_utils import evaluate_metric
from examples.example_utils import init_input_data
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

metric_dict = {}

group = os.listdir('D:\WORK\Repo\Industiral\IndustrialTS\data')

model_dict = {
    'eigen_basis_basic': PipelineBuilder().add_node(
        'eigen_basis',
        params={'low_rank_approximation': False}).add_node(
        'quantile_extractor',
        params={'window_size': 10, 'window_mode': False}).add_node(
        'logit'),

    'eigen_basis_advanced': PipelineBuilder().add_node(
        'eigen_basis', params={'low_rank_approximation': True}).add_node(
        'quantile_extractor', params={'window_size': 10, 'window_mode': False}).add_node(
        'logit')}

datasets_bad_f1 = [
    'EOGVerticalSignal',
    'ScreenType',
    'CricketY',
    'ElectricDevices',
    'Lightning7'
]

datasets_good_f1 = [
    'Car',
    'ECG5000',
    "Beef"
]

datasets_good_roc = [
    'Chinatown',
    'Computers',
    'Earthquakes',
    'Ham',
    'ECG200',
    'ECGFiveDays',
    'MiddlePhalanxOutlineCorrect',
    'MoteStrain',
    'TwoLeadECG'
]

if __name__ == "__main__":
    for dataset_name in group:
        try:
            train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])
            with IndustrialModels():
                for model in model_dict.keys():
                    pipeline = model_dict[model].build()
                    pipeline.fit(input_data)
                    features = pipeline.predict(val_data, 'labels').predict
                    metric = evaluate_metric(target=test_data[1], prediction=features)
                    metric_dict.update({f'{dataset_name}_{model}': metric})
                    print(f'{dataset_name}_{model} - {metric}')
        except Exception:
            print(f'{dataset_name} doesnt exist')
    print(metric_dict)

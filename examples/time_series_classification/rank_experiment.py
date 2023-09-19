import os

import numpy as np
import matplotlib.pyplot as plt
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from sklearn.metrics import f1_score, roc_auc_score
from examples.fedot.fedot_ex import init_input_data
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

# def visualise_and_save():
#     for class_number in np.unique(train_data[1]):
#         for basis_name, basis in zip(['basis_before_power_iterations', 'basis_after_power_iterations'],
#                                      [basis_1d_raw, basis_1d_approx]):
#             class_idx = np.where(train_data[1] == class_number)[0]
#             class_slice = np.take(basis, class_idx, 0)
#             pd.DataFrame(np.median(class_slice, axis=0)).T.plot()
#             # plt.show()
#             plt.savefig(f'{dataset_name}/{basis_name}_{class_number}_median_component.png', bbox_inches='tight')
#             # plt.title(f'mean_{basis_name}_components_for_{class_number}_class')
#     rank_distrib = pd.DataFrame([rank_distribution_befor, rank_distribution_after]).T
#     rank_distrib.columns = ['HT_approach',
#                             'Proposed_approach']
#     rank_distrib.plot(kind='kde')
#     # plt.show()
#     rank_dispersion_ht = np.round(rank_distrib['HT_approach'].std(), 3)
#     rank_dispersion_new = np.round(rank_distrib['Proposed_approach'].std(), 3)
#     plt.savefig(f'{dataset_name}/rank_distrib. '
#                 f'Classical_rank_{low_rank_befor}_std_{rank_dispersion_ht}.'
#                 f'New_{low_rank_after}_std_{rank_dispersion_new}.png', bbox_inches='tight')
#     rank_distrib['classes'] = train_data[1]

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
    "Beef",
    # 'Phoneme',
    # 'Meat',
    # 'RefrigerationDevices'
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

group = os.listdir('D:\WORK\Repo\Industiral\IndustrialTS\data')

model_dict = {
    'eigen_basis_basic': PipelineBuilder().add_node('eigen_basis', params={'low_rank_approximation': False}).add_node(
        'quantile_extractor',
        params={'window_size': 10,
                'window_mode': False}).add_node(
        'logit'),
    'eigen_basis_advanced': PipelineBuilder().add_node('eigen_basis', params={'low_rank_approximation': True}).add_node(
        'quantile_extractor',
        params={'window_size': 10,
                'window_mode': False}).add_node(
        'logit')}
metric_dict = {}


def evaluate_metric(target, prediction):
    try:
        if len(np.unique(target)) > 2:
            metric = f1_score(target, prediction, average='weighted')
        else:
            metric = roc_auc_score(target, prediction, average='weighted')
    except Exception:
        metric = 0
    return metric


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

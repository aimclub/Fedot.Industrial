import os

import numpy as np
import matplotlib.pyplot as plt
from fedot.api.main import Fedot
from sklearn.metrics import f1_score, roc_auc_score
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def extract_features(train_data, bss):
    basis_1d_raw = bss._transform(train_data[0])
    feature_train = stats_model.transform(basis_1d_raw)
    return feature_train, bss


def evaluate_model(feature_train, bss, test_data, model_type: str = 'MLP'):
    if len(np.unique(test_data[1])) > 2:
        metric_name = 'f1'
    else:
        metric_name = 'roc_auc'

    if model_type == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                            random_state=42)
    else:
        clf = Fedot(
            # available_operations=['fast_ica', 'scaling','normalization',
            #                               'xgboost',
            #                               'rf',
            #                               'logit',
            #                               'mlp',
            #                               'knn',
            #                               'pca'],
            metric=metric_name, timeout=10, problem='classification', n_jobs=6)

    scaler = StandardScaler()
    scaler.fit(feature_train)
    feature_train = scaler.transform(feature_train)
    clf.fit(feature_train, train_data[1])
    basis_1d_raw = bss._transform(test_data[0])
    test_feature = stats_model.transform(basis_1d_raw)
    test_feature = scaler.transform(test_feature)
    if len(np.unique(test_data[1])) > 2:
        metric = f1_score(test_data[1], clf.predict(test_feature), average='weighted')
    else:
        metric = roc_auc_score(test_data[1], clf.predict(test_feature), average='weighted')
    return metric, test_feature


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


if __name__ == "__main__":

    datasets_bad_f1 = [
        # 'EOGVerticalSignal',
        # 'ScreenType',
        # 'CricketY',
        # 'ElectricDevices',
        'Lightning7'
    ]

    datasets_good_f1 = [
        'Car',
        'ECG5000',
        "Beef",
        #     'Phoneme',
        'Meat',
        # 'RefrigerationDevices'
    ]

    datasets_good_roc = [
        # 'Chinatown',
        'Computers',
        # 'Earthquakes',
        'Ham',
        'ECG200',
        'ECGFiveDays'
        # 'MiddlePhalanxOutlineCorrect',
        # 'MoteStrain',
        # 'TwoLeadECG'
    ]
    # node_scaling = PipelineNode('scaling')
    # node_final = PipelineNode('rf', nodes_from=[node_scaling])
    # rf_model = Pipeline(node_final)

    datasets_bad_roc = [
        'Lightning2',
        # 'WormsTwoClass',
        # 'DistalPhalanxOutlineCorrect'
    ]

    stats_model = QuantileExtractor({'window_mode': False, 'window_size': 5, 'use_cache': False, 'n_jobs': 4})
    for group in [
        datasets_bad_f1,
        datasets_good_f1,
        datasets_good_roc,
        datasets_bad_roc
    ]:

        for dataset_name in group:

            industrial = FedotIndustrial(task='ts_classification',
                                         dataset=dataset_name,
                                         strategy='fedot_preset',
                                         branch_nodes=[
                                             # 'fourier_basis',
                                             # 'wavelet_basis',
                                             'data_driven_basis'
                                         ],
                                         tuning_iterations=30,
                                         tuning_timeout=15,
                                         use_cache=False,
                                         timeout=5,
                                         n_jobs=6,
                                         )
            try:
                os.makedirs(f'./{dataset_name}')
            except Exception:
                _ = 1

            train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
            # bss = EigenBasisImplementation({'sv_selector': 'median', 'window_size': 20})
            # bss.low_rank_approximation = False
            # train_feature, bss = extract_features(train_data, bss)
            # f1_HT, test_feature = evaluate_model(train_feature, bss, test_data,model_type='Auto')

            bss = EigenBasisImplementation({'sv_selector': 'median', 'window_size': 20})
            bss.low_rank_approximation = True
            bss.SV_threshold = None
            train_feature, bss = extract_features(train_data, bss)
            f1_PI, test_feature_PI = evaluate_model(train_feature, bss, test_data, model_type='Auto')
            print(f'Dataset-{dataset_name}')
            # print(f'HT_metric-{f1_HT}')
            print(f'PI_metric-{f1_PI}')
    _ = 1

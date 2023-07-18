import os

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
import matplotlib.pyplot as plt
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.models.statistical.StatsExtractor import StatsExtractor
from fedot_ind.core.operation.transformation.basis.data_driven import DataDrivenBasisImplementation

if __name__ == "__main__":

    datasets_bad_f1 = [
        # 'EOGVerticalSignal',
        # 'ScreenType',
        # 'CricketY',
        # 'ElectricDevices',
        'Lightning7'
    ]

    datasets_good_f1 = [
        # 'Car',
        # 'ECG5000',
        # "Beef"
        #     'Phoneme',
        # 'Meat',
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

    datasets_bad_roc = [
        'Lightning2',
        # 'WormsTwoClass',
        # 'DistalPhalanxOutlineCorrect'
    ]
    stats_model = StatsExtractor({'window_mode': False, 'window_size': 5, 'use_cache': False, 'n_jobs': 4})
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
                                         tuning_iterations=10,
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
            bss = DataDrivenBasisImplementation({'sv_selector': 'median', 'window_size': 20})
            bss.low_rank_approximation = False
            basis_1d_raw = bss._transform(train_data[0])
            rank_distribution_befor = bss.rank_distribution
            low_rank_befor = bss.SV_threshold
            bss.low_rank_approximation = True
            bss.SV_threshold = None
            basis_1d_approx = bss._transform(train_data[0])
            rank_distribution_after = bss.rank_distribution
            low_rank_after = bss.SV_threshold

            HT_feature = stats_model.transform(basis_1d_raw)
            for class_number in np.unique(train_data[1]):
                for basis_name, basis in zip(['basis_before_power_iterations', 'basis_after_power_iterations'],
                                             [basis_1d_raw, basis_1d_approx]):
                    class_idx = np.where(train_data[1] == class_number)[0]
                    class_slice = np.take(basis, class_idx, 0)
                    pd.DataFrame(np.median(class_slice, axis=0)).T.plot()
                    # plt.show()
                    plt.savefig(f'{dataset_name}/{basis_name}_{class_number}_median_component.png', bbox_inches='tight')
                    # plt.title(f'mean_{basis_name}_components_for_{class_number}_class')
            rank_distrib = pd.DataFrame([rank_distribution_befor, rank_distribution_after]).T
            rank_distrib.columns = ['HT_approach',
                                    'Proposed_approach']
            rank_distrib.plot(kind='kde')
            # plt.show()
            rank_dispersion_ht = np.round(rank_distrib['HT_approach'].std(), 3)
            rank_dispersion_new = np.round(rank_distrib['Proposed_approach'].std(), 3)
            plt.savefig(f'{dataset_name}/rank_distrib. '
                        f'Classical_rank_{low_rank_befor}_std_{rank_dispersion_ht}.'
                        f'New_{low_rank_after}_std_{rank_dispersion_new}.png', bbox_inches='tight')
            rank_distrib['classes'] = train_data[1]
    _ = 1

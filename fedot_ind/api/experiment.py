import os

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader

if __name__ == "__main__":

    # datasets_bad_f1 = [
    #         'EOGVerticalSignal',
    #     'ScreenType',
    #     'CricketY',
    #     'ElectricDevices',
    #     'Lightning7'
    # ]

    # datasets_good_f1 = [
    #     'Car',
    # 'ECG5000',
        # 'Phoneme',
        # 'Meat',
        # 'RefrigerationDevices'
    # ]

    datasets_good_roc = [
    #     # 'Chinatown',
    # 'Earthquakes',
    # 'Ham',
    'ECG200',
    # 'MiddlePhalanxOutlineCorrect',
    # 'MoteStrain',
    # 'TwoLeadECG'
    ]

    # datasets_bad_roc = [
    #     'Lightning2',
    #     'WormsTwoClass',
    #     'DistalPhalanxOutlineCorrect'
    # ]

    for group in [
        # datasets_bad_f1,
        # datasets_good_f1,
        datasets_good_roc,
        # datasets_bad_roc
    ]:

        for dataset_name in group:
            experiment = 'good_f1'

            industrial = FedotIndustrial(task='ts_classification',
                                         dataset=dataset_name,
                                         strategy='fedot_preset',
                                         branch_nodes=[
                                             # 'fourier_basis',
                                             # 'wavelet_basis',
                                             'data_driven_basis'
                                         ],
                                         tuning_iterations=3,
                                         use_cache=False,
                                         timeout=5,
                                         n_jobs=2,
                                         )

            train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
            model = industrial.fit(features=train_data[0], target=train_data[1])
            labels = industrial.predict(features=test_data[0],
                                        target=test_data[1])
            probs = industrial.predict_proba(features=test_data[0],
                                             target=test_data[1])
            metric = industrial.get_metrics(target=test_data[1],
                                            metric_names=['f1', 'roc_auc'])
            for pred, kind in zip([labels, probs], ['labels', 'probs']):
                industrial.save_predict(predicted_data=pred, kind=kind)

            industrial.save_metrics(metrics=metric)

            # load ranks array and save it to json file with dataset name
            # ranks_path = '/Users/technocreep/Desktop/Working-Folder/fedot-industrial/Fedot.Industrial/fedot_ind/results_of_experiments/fedot_preset/ranks.npy'
            #
            # import numpy as np
            # ranks = np.load(ranks_path)
            # import json
            #
            # # dataset_name = 'Lightning7'
            # ranks_dict = {dataset_name: ranks.tolist()}
            #
            # archived_ranks_path = '/Users/technocreep/Desktop/Working-Folder/fedot-industrial/Fedot.Industrial/fedot_ind/results_of_experiments/fedot_preset/rank.json'
            #
            # if not os.path.exists(archived_ranks_path):
            #     with open(archived_ranks_path, 'w') as f:
            #         json.dump(ranks_dict, f)
            # else:
            #
            #     with open(archived_ranks_path, 'r') as f:
            #         archive = json.load(f)
            #
            #     archive.update(ranks_dict)
            #     with open(archived_ranks_path, 'w') as f:
            #         json.dump(archive, f)
            #
            # # delete ranks_path
            # os.remove(ranks_path)
            #
            # _ = 1

    # visualisation of distribution of ranks for each dataset

    # import json
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    # from collections import defaultdict
    #
    # archived_ranks_path = '/Users/technocreep/Desktop/Working-Folder/fedot-industrial/Fedot.Industrial/fedot_ind/results_of_experiments/fedot_preset/rank.json'
    #
    # with open(archived_ranks_path, 'r') as f:
    #     archive = json.load(f)
    #
    # ranks_dict = defaultdict(list)
    # for dataset_name, ranks in archive.items():
    #     ranks_dict[dataset_name] = ranks['train'] + ranks['test']
    #
    # ranks_df = pd.DataFrame(ranks_dict)
    # ranks_df = ranks_df.melt()
    # ranks_df.columns = ['dataset', 'rank']
    #
    # plt.figure(figsize=(20, 10))
    # sns.boxplot(x='dataset', y='rank', data=ranks_df)
    # plt.xticks(rotation=90)
    # plt.show()
    #
    # import seaborn as sns
    # for ds in archive.keys():
    #     train_ds = np.array(archive[ds]['train'])
    #     test_ds = np.array(archive[ds]['test'])
    #
    #     sns.displot(train_ds, bins=82, kde=True).set(title=f'{ds} train')
    #     sns.displot(test_ds, bins=82, kde=True).set(title=f'{ds} test')
    #     plt.show()

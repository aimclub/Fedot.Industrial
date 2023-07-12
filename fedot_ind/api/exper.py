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
        'Chinatown',
    # 'Earthquakes',
    # 'Ham',
    # 'ECG200',
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

            industrial = FedotIndustrial(task='ts_classification',
                                         dataset=dataset_name,
                                         strategy='fedot_preset',
                                         branch_nodes=[
                                             # 'fourier_basis',
                                             # 'wavelet_basis',
                                             'data_driven_basis'
                                         ],
                                         tuning_iterations=10,
                                         tuning_timeout=2,
                                         use_cache=False,
                                         timeout=1,
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

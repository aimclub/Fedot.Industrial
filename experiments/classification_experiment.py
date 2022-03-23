from experiments.run.SSARunner import SSARunner
from experiments.run.utils import read_tsv

if __name__ == '__main__':
    dict_of_dataset = {
        # 'Beef': read_tsv('Beef'),
        # 'DodgerLoopDay': read_tsv('DodgerLoopDay'),
        # 'ElectricDevices': read_tsv('ElectricDevices'),
        'EthanolLevel': read_tsv('EthanolLevel'),
        # 'Lightning7': read_tsv('Lightning7'),
        # 'Plane': read_tsv('Plane'),
        'PowerCons': read_tsv('PowerCons'),
        'Rock': read_tsv('Rock'),
        'ShapesAll': read_tsv('ShapesAll'),
        # 'Trace': read_tsv('Trace'),

        # 'LSST': read_tsv('LSST'),
        # 'ScreenType': read_tsv('ScreenType'),
        # 'EigenWorms': read_tsv('EigenWorms'),
        # 'AsphaltRegularity': read_tsv('AsphaltRegularity'),

        # 'Herring': read_tsv('Herring')
    }
    if __name__ == '__main__':
        dict_of_win_list = {
            # 'Beef': [47, 94, 141],
            # 'DodgerLoopDay': [29, 58, 86],
            # 'ElectricDevices': [10, 19, 29],
            'EthanolLevel': [175, 350, 525],
            # 'Lightning7': [32, 64, 96],
            # 'Plane': [14, 29, 43],
            'PowerCons': [14, 29, 43],
            'Rock': [284, 569, 853],
            'ShapesAll': [51, 102, 154],
            # 'Trace': [28, 55, 83],

            # test dataset
            # 'Herring': [51
            #     # , 102, 154
            #             ],

            # NOT READY YET
            # 'LSST': 4,
            # 'ScreenType': 72,
            # 'EigenWorms': 1798,
            # 'ArticularyWordRecognition': 14
        }

        list_of_dataset = [
            # 'Beef',
            # 'DodgerLoopDay',
            # 'ElectricDevices',
            'EthanolLevel',
            # 'Lightning7',
            # 'Plane',
            'PowerCons',
            'Rock',
            'ShapesAll',
            # 'Trace',

            # # test dataset
            # 'Herring',

            # NOT READY YET
            # 'LSST',
            # 'ScreenType',
            # 'EigenWorms',
            # 'ArticularyWordRecognition'
        ]

    fedot_params = {'problem': 'classification',
                    'seed': 42,
                    'timeout': 5,
                    'composer_params': {'max_depth': 10,
                                        'max_arity': 4,
                                        'available_operations': ['resample', 'scaling', 'simple_imputation', 'rf',
                                                                 'isolation_forest_class', 'lgbm',
                                                                 'pca', 'logit', 'normalization', 'mlp',
                                                                 'one_hot_encoding', 'knn']},
                    'verbose_level': 2,
                    'n_jobs': 7}

    runner = SSARunner(list_of_dataset,
                       launches=3,
                       fedot_params=fedot_params)

    models = runner.run_experiment(dict_of_dataset,
                                   dict_of_win_list)

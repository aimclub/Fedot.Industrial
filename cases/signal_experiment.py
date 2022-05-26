from cases.run.SignalRunner import SignalRunner
from cases.run.utils import read_tsv

if __name__ == '__main__':
    dict_of_dataset = {
        'ElectricDevices': read_tsv('ElectricDevices'),
        'Earthquakes': read_tsv('Earthquakes'),
        'Beef': read_tsv('Beef'),
        'Lightning7': read_tsv('Lightning7'),
    }

    dict_of_wavelet_list = {'ElectricDevices': ['db5', 'sym5', 'coif5', 'bior2.4'],
                            'Earthquakes': ['db5', 'sym5', 'coif5', 'bior2.4'],
                            'Lightning7': ['db5', 'sym5', 'coif5', 'bior2.4'],
                            'Beef': ['db5', 'sym5', 'coif5', 'bior2.4']
                            }

    list_of_dataset = [
        #'Beef',
                       #'ElectricDevices',
                       'Earthquakes',
                       #'Lightning7'
                       ]

    fedot_params = {'problem': 'classification',
                    'seed': 42,
                    'timeout': 5,
                    'composer_params': {'max_depth': 10,
                                        'max_arity': 4,
                                        'cv_folds': 3,
                                        'stopping_after_n_generation': 20
                                        },
                    'verbose_level': 2,
                    'n_jobs': 4}

    runner = SignalRunner(list_of_dataset,
                          launches=1,
                          fedot_params=fedot_params)

    models = runner.run_experiment(dict_of_dataset=dict_of_dataset,dict_of_win_list=dict_of_wavelet_list,method='ww')

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from cases.run.QuantileRunner import StatsRunner
from cases.run.utils import read_tsv

if __name__ == '__main__':
    dict_of_dataset = {
        'ElectricDevices': read_tsv('ElectricDevices'),
        'Earthquakes': read_tsv('Earthquakes'),
        'Beef': read_tsv('Beef'),
        'Lightning7': read_tsv('Lightning7'),
        'EthanolLevel': read_tsv('EthanolLevel')
    }

    dict_of_wavelet_list = {
        'ElectricDevices': ['db5', 'sym5', 'coif5', 'bior2.4'],
        'EthanolLevel': ['db5', 'sym5', 'coif5', 'bior2.4'],
        'Earthquakes': ['db5', 'sym5', 'coif5', 'bior2.4'],
        'Lightning7': ['db5', 'sym5', 'coif5', 'bior2.4'],
        'Beef': ['db5', 'sym5', 'coif5', 'bior2.4'],
    }

    list_of_dataset = [
        'ElectricDevices',
        'Earthquakes',
        'Beef',
        'Lightning7',
        'EthanolLevel'
    ]

    fedot_params = {'problem': 'classification',
                    'seed': 42,
                    'timeout': 15,
                    'composer_params': {'max_depth': 10,
                                        'max_arity': 4,
                                        'cv_folds': 3,
                                        'stopping_after_n_generation': 20
                                        },
                    'verbose_level': 2,
                    'n_jobs': 6}

    runner = StatsRunner(list_of_dataset,
                         launches=3,
                         fedot_params=fedot_params,
                         static_booster=False)

    models = runner.run_experiment(dict_of_dataset,
                                   dict_of_wavelet_list)

from experiments.run.SSARunner import SSARunner
from experiments.run.utils import read_tsv

if __name__ == '__main__':
    dict_of_dataset = {
        'Herring': read_tsv('Herring'),
        'Haptics': read_tsv('Haptics'),
        'DodgerLoopDay': read_tsv('DodgerLoopDay'),
        'Earthquakes': read_tsv('Earthquakes'),
        'FordA': read_tsv('FordA'),
        'FordB': read_tsv('FordB'),
        'Plane': read_tsv('Plane'),
        'Trace': read_tsv('Trace'),
        'ItalyPowerDemand': read_tsv('ItalyPowerDemand'),
        'Lightning7': read_tsv('Lightning7'),
    }

    dict_of_win_list = {'gunpoint': 30,
                        'basic_motions': 10,
                        'arrow_head': 50,
                        'osuleaf': 90,
                        'ItalyPowerDemand': [3, 6, 9],
                        'unit_test': 3,
                        'Herring': [48, 128, 170],
                        'Haptics': [110, 220, 330],
                        'DodgerLoopDay': [28, 56, 84],
                        'Earthquakes': [48, 128, 170],
                        'FordA': [50, 100, 150],
                        'FordB': [50, 100, 150],
                        'Plane': [14, 28, 42],
                        'Trace': [27, 54, 81],
                        'Lightning7': [32, 64, 96]
                        }

    list_of_dataset = [
        'ItalyPowerDemand',
        'Herring',
        'Haptics',
        'DodgerLoopDay',
        'Earthquakes',
        'FordA',
        'FordB',
        'Plane',
        'Trace',
        'Lightning7'
    ]

    fedot_params = {'problem': 'classification',
                    'seed': 42,
                    'timeout': 20,
                    'composer_params': {'max_depth': 10,
                                        'max_arity': 4,
                                        'cv_folds': 3
                                        },
                    'verbose_level': 2,
                    'n_jobs': 4}

    runner = SSARunner(list_of_dataset,
                       launches=3,
                       fedot_params=fedot_params)

    models = runner.run_experiment(dict_of_dataset,
                                   dict_of_win_list)

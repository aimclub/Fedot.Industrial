from experiments.run.TopologicalRunner import TopologicalRunner
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
                        'ItalyPowerDemand': 3,
                        'unit_test': 3,
                        'Herring': 170,
                        'Haptics': 300,
                        'DodgerLoopDay': 80,
                        'Earthquakes': 128,
                        'FordA': 125,
                        'FordB': 125,
                        'Plane': 48,
                        'Trace': 90,
                        'Lightning7': 100
                        }

    list_of_dataset = ['ItalyPowerDemand',
                       # 'Herring',
                       # 'Haptics',
                       # 'DodgerLoopDay',
                       #'Earthquakes',
                       # 'FordA',
                       # 'FordB',
                       # 'Plane',
                       # 'Trace',
                       # 'Lightning7'
                       ]

    topological_params = {
        'max_simplex_dim': 2,
        'epsilon': 100,
        'persistance_params': None,
        'window_length': 6}

    runner = TopologicalRunner(topological_params=topological_params,
                               list_of_dataset=list_of_dataset,
                               launches=1)
    models = runner.run_experiment(dict_of_dataset,
                                   dict_of_win_list)

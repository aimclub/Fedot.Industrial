from cases.run.QuantileRunner import StatsRunner
from cases.run.TopologicalRunner import TopologicalRunner
from cases.run.utils import read_tsv

if __name__ == '__main__':
    dict_of_dataset = {
        'Beef': read_tsv('Beef')
    }

    dict_of_win_list = {'Beef': 50}

    list_of_dataset = [
        'Beef',
    ]

    topological_params = {
        'max_simplex_dim': 2,
        'epsilon': 100,
        'persistance_params': None,
        'window_length': 6}

    fedot_params = {'problem': 'classification',
                    'seed': 42,
                    'timeout': 15,
                    'composer_params': {'max_depth': 10,
                                        'max_arity': 4,
                                        'cv_folds': 3,
                                        'stopping_after_n_generation': 20
                                        },
                    'verbose_level': 2,
                    'n_jobs': 4}

    runner_stats = StatsRunner(list_of_dataset,
                               launches=3,
                               fedot_params=fedot_params)

    stats_features = runner_stats.extract_features(dataset='Beef',
                                                   dict_of_dataset=dict_of_dataset)

    runner_topo = TopologicalRunner(topological_params=topological_params,
                                    list_of_dataset=list_of_dataset,
                                    launches=1,
                                    fedot_params=fedot_params)

    topo_features = runner_topo.extract_features(dataset='Beef',
                                                 dict_of_dataset=dict_of_dataset,
                                                 dict_of_extra_params=dict_of_win_list
                                                 )

from enum import Enum


class GeneratorParams(Enum):
    # quantile = {'window_mode': True,
    #             'use_cache': False,
    #             'window_size': 20}

    statistical = {'window_mode': False,
                   'var_threshold': 0,
                   'use_cache': False}

    recurrence = {'use_cache': False,
                  'threshold_baseline': [1, 5, 10, 15, 20, 25, 30],
                  'min_signal_ratio': 0.65,
                  'max_signal_ratio': 0.75,
                  'rec_metric': 'euclidean'}

    topological = {'use_cache': False,
                   'max_te_dimension': 5,
                   'max_te_time_delay': 2,
                   'stride': 1}

    spectral = {'window_sizes': [],
                'window_mode': True,
                'use_cache': False,
                'correlation_level': 0.8,
                'combine_eigenvectors': False}

    ensemble = {'list_of_generators': {},
                'use_cache': False}

    window_spectral = {'window_mode': True,
                       'window_sizes': [],
                       'use_cache': False}

    wavelet = {'use_cache': False,
               'wavelet': 'mexh'}

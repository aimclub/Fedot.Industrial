from enum import Enum


class GeneratorParams(Enum):

    quantile = {'window_mode': False,
                'use_cache': False}

    recurrence = {'use_cache': False,
                  'threshold_baseline': [1, 5, 10, 15, 20, 25, 30],
                  'min_signal_ratio': 0.65,
                  'max_signal_ratio': 0.75,
                  'rec_metric': 'euclidean'}

    topological = {'use_cache': False,
                   'te_dimension': None,
                   'te_time_delay': None}

    spectral = {'window_sizes': [],
                'window_mode': True,
                'use_cache': False,
                'correlation_level': 0.8,
                'combine_eigenvectors': False}

    ensemble = {'list_of_generators': { },
                'use_cache': False}

    window_quantile = {'window_mode': True,
                       'use_cache': False}

    window_spectral = {'window_mode': True,
                       'window_sizes': [ ],
                       'use_cache': False}

    wavelet = {'wavelet_types': [ 'db5', 'sym5', 'coif5', 'bior2.4' ],
               'use_cache': False}

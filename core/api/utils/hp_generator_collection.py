from enum import Enum


class GeneratorParams(Enum):

    quantile = {'window_mode': False}
    recurrence = {}
    topological = {}
    spectral = {'window_sizes': []}
    ensemble = {'list_of_generators': { }}
    window_quantile = {'window_mode': True}
    window_spectral = {'window_mode': True, 'window_sizes': [ ]}
    wavelet = {'wavelet_types': [ 'db5', 'sym5', 'coif5', 'bior2.4' ]}

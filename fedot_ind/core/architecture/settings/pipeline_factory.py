from enum import Enum

from fedot_ind.core.models.detection.probalistic.kalman import UnscentedKalmanFilter
from fedot_ind.core.models.detection.subspaces.func_pca import FunctionalPCA
from fedot_ind.core.models.detection.subspaces.sst import SingularSpectrumTransformation
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.recurrence.RecurrenceExtractor import RecurrenceExtractor
from fedot_ind.core.models.signal.SignalExtractor import SignalExtractor
from fedot_ind.core.models.topological.TopologicalExtractor import TopologicalExtractor
from fedot_ind.core.operation.transformation.basis.data_driven import DataDrivenBasisImplementation
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation


class BasisTransformations(Enum):
    datadriven = DataDrivenBasisImplementation
    wavelet = WaveletBasisImplementation
    Fourier = FourierBasisImplementation


class FeatureGenerator(Enum):
    quantile = QuantileExtractor
    wavelet = SignalExtractor
    topological = TopologicalExtractor
    recurrence = RecurrenceExtractor


class MlModel(Enum):
    functional_pca = FunctionalPCA
    kalman_filter = UnscentedKalmanFilter
    sst = SingularSpectrumTransformation


class KernelFeatureGenerator(Enum):
    statistical = [{'feature_generator_type': 'quantile',
                    'feature_hyperparams': {
                        'window_mode': True,
                        'window_size': 5
                    }
                    },
                   {'feature_generator_type': 'quantile',
                    'feature_hyperparams': {
                        'window_mode': True,
                        'window_size': 10
                    }
                    },
                   {'feature_generator_type': 'quantile',
                    'feature_hyperparams': {
                        'window_mode': True,
                        'window_size': 20
                    }
                    },
                   {'feature_generator_type': 'quantile',
                    'feature_hyperparams': {
                        'window_mode': True,
                        'window_size': 30
                    }
                    },
                   {'feature_generator_type': 'quantile',
                    'feature_hyperparams': {
                        'window_mode': True,
                        'window_size': 40
                    }
                    }
                   ]
    wavelet = [
        {'feature_generator_type': 'wavelet',
         'feature_hyperparams': {
             'wavelet': "mexh",
             'n_components': 2
         }}
        # ,
        # {'feature_generator_type': 'wavelet',
        #  'feature_hyperparams': {
        #      'wavelet': "haar",
        #      'n_components': 2
        #  }}
        ,
        # {'feature_generator_type': 'wavelet',
        #  'feature_hyperparams': {
        #      'wavelet': "dmey",
        #      'n_components': 2
        #  }
        #  },
        {'feature_generator_type': 'wavelet',
         'feature_hyperparams': {
             'wavelet': "gaus3",
             'n_components': 2
         }
         },
        {'feature_generator_type': 'wavelet',
         'feature_hyperparams': {
             'wavelet': "morl",
             'n_components': 2
         }
         }
    ]
    recurrence = []
    topological = []

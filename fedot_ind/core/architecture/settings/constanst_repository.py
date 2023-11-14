import math
from enum import Enum
from multiprocessing import cpu_count

import pywt
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.models.quantile.stat_features import *
from fedot_ind.core.models.topological.topofeatures import *
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix


def beta_thr(beta):
    return 0.56 * np.power(beta, 3) - 0.95 * np.power(beta, 2) + 1.82 * beta + 1.43


class ComputationalConstant(Enum):
    CPU_NUMBERS = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1


class DataTypeConstant(Enum):
    MULTI_ARRAY = DataTypesEnum.image
    MATRIX = DataTypesEnum.table
    TRAJECTORY_MATRIX = HankelMatrix


class FeatureConstant(Enum):
    STAT_METHODS = {'mean_': np.mean,
                    'median_': np.median,
                    'std_': np.std,
                    'max_': np.max,
                    'min_': np.min,
                    'q5_': q5,
                    'q25_': q25,
                    'q75_': q75,
                    'q95_': q95,
                    # 'sum_': np.sum,
                    # 'dif_': diff
                    }

    STAT_METHODS_GLOBAL = {
        'skewness_': skewness,
        'kurtosis_': kurtosis,
        'n_peaks_': n_peaks,
        'slope_': slope,
        'ben_corr_': ben_corr,
        'interquartile_range_': interquartile_range,
        'energy_': energy,
        'cross_rate_': zero_crossing_rate,
        'autocorrelation_': autocorrelation,
        # 'base_entropy_': base_entropy,
        'shannon_entropy_': shannon_entropy,
        'ptp_amplitude_': ptp_amp,
        'mean_ptp_distance_': mean_ptp_distance,
        'crest_factor_': crest_factor,
        'mean_ema_': mean_ema,
        'mean_moving_median_': mean_moving_median,
        'hjorth_mobility_': hjorth_mobility,
        'hjorth_complexity_': hjorth_complexity,
        'hurst_exponent_': hurst_exponent,
        'petrosian_fractal_dimension_': pfd,
    }

    PERSISTENCE_DIAGRAM_FEATURES = {'HolesNumberFeature': HolesNumberFeature(),
                                    'MaxHoleLifeTimeFeature': MaxHoleLifeTimeFeature(),
                                    'RelevantHolesNumber': RelevantHolesNumber(),
                                    'AverageHoleLifetimeFeature': AverageHoleLifetimeFeature(),
                                    'SumHoleLifetimeFeature': SumHoleLifetimeFeature(),
                                    'PersistenceEntropyFeature': PersistenceEntropyFeature(),
                                    'SimultaneousAliveHolesFeature': SimultaneousAliveHolesFeature(),
                                    'AveragePersistenceLandscapeFeature': AveragePersistenceLandscapeFeature(),
                                    'BettiNumbersSumFeature': BettiNumbersSumFeature(),
                                    'RadiusAtMaxBNFeature': RadiusAtMaxBNFeature()}

    PERSISTENCE_DIAGRAM_EXTRACTOR = PersistenceDiagramsExtractor(takens_embedding_dim=1,
                                                                 takens_embedding_delay=2,
                                                                 homology_dimensions=(0, 1),
                                                                 parallel=False)
    DISCRETE_WAVELETS = pywt.wavelist(kind='discrete')
    CONTINUOUS_WAVELETS = pywt.wavelist(kind='continuous')
    WAVELET_SCALES = [2, 4, 10, 20]
    SINGULAR_VALUE_MEDIAN_THR = 2.58
    SINGULAR_VALUE_BETA_THR = beta_thr


class FedotOperationConstant(Enum):
    EXCLUDED_OPERATION = ['fast_ica']

    AVAILABLE_CLS_OPERATIONS = [
        'rf',
        'logit',
        'scaling',
        'normalization',
        'pca',
        'catboost',
        'svc',
        'knn',
        'kernel_pca',
        'isolation_forest_class']

    AVAILABLE_REG_OPERATIONS = ['rfr',
                                'ridge',
                                'scaling',
                                'normalization',
                                'pca',
                                'catboostreg',
                                'xgbreg',
                                'svr',
                                'dtreg',
                                'treg',
                                'knnreg',
                                'kernel_pca',
                                'isolation_forest_reg',
                                'rfe_lin_reg',
                                'rfe_non_lin_reg']


class ModelCompressionConstant(Enum):
    ENERGY_THR = [0.9, 0.95, 0.99, 0.999]
    DECOMPOSE_MODE = 'channel'
    FORWARD_MODE = 'one_layer'
    HOER_LOSS = 0.1
    ORTOGONAL_LOSS = 10
    MODELS_FROM_LENGHT = {
        122: 'ResNet18',
        218: 'ResNet34',
        320: 'ResNet50',
        626: 'ResNet101',
        932: 'ResNet152',
    }


STAT_METHODS = FeatureConstant.STAT_METHODS.value
STAT_METHODS_GLOBAL = FeatureConstant.STAT_METHODS_GLOBAL.value
PERSISTENCE_DIAGRAM_FEATURES = FeatureConstant.PERSISTENCE_DIAGRAM_FEATURES.value
PERSISTENCE_DIAGRAM_EXTRACTOR = FeatureConstant.PERSISTENCE_DIAGRAM_EXTRACTOR.value
DISCRETE_WAVELETS = FeatureConstant.DISCRETE_WAVELETS.value
CONTINUOUS_WAVELETS = FeatureConstant.CONTINUOUS_WAVELETS.value
WAVELET_SCALES = FeatureConstant.WAVELET_SCALES.value
SINGULAR_VALUE_MEDIAN_THR = FeatureConstant.SINGULAR_VALUE_MEDIAN_THR.value
SINGULAR_VALUE_BETA_THR = FeatureConstant.SINGULAR_VALUE_BETA_THR

AVAILABLE_REG_OPERATIONS = FedotOperationConstant.AVAILABLE_REG_OPERATIONS.value
AVAILABLE_CLS_OPERATIONS = FedotOperationConstant.AVAILABLE_CLS_OPERATIONS.value
EXCLUDED_OPERATION = FedotOperationConstant.EXCLUDED_OPERATION.value

CPU_NUMBERS = ComputationalConstant.CPU_NUMBERS.value

MULTI_ARRAY = DataTypeConstant.MULTI_ARRAY.value
MATRIX = DataTypeConstant.MATRIX.value
TRAJECTORY_MATRIX = DataTypeConstant.TRAJECTORY_MATRIX.value

ENERGY_THR = ModelCompressionConstant.ENERGY_THR.value
DECOMPOSE_MODE = ModelCompressionConstant.DECOMPOSE_MODE.value
FORWARD_MODE = ModelCompressionConstant.FORWARD_MODE.value
HOER_LOSS = ModelCompressionConstant.HOER_LOSS.value
ORTOGONAL_LOSS = ModelCompressionConstant.ORTOGONAL_LOSS.value
MODELS_FROM_LENGHT = ModelCompressionConstant.MODELS_FROM_LENGHT.value
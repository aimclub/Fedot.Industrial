from copy import deepcopy

import numpy as np
from hyperopt import hp

from fedot_ind.core.repository.forecasting_registry import FORECASTING_MODEL_ALIASES

NESTED_PARAMS_LABEL = 'nested_label'

industrial_search_space = {
    'eigen_basis':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
         'rank_regularization': {'hyperopt-dist': hp.choice, 'sampling-scope': [
             ['hard_thresholding', 'explained_dispersion']]},
         'decomposition_type': {'hyperopt-dist': hp.choice, 'sampling-scope': [['svd', 'random_svd']]}
         },
    'wavelet_basis':
        {'n_components': {'hyperopt-dist': hp.uniformint, 'sampling-scope': [2, 10]},
         'wavelet': {'hyperopt-dist': hp.choice,
                     'sampling-scope': [['mexh', 'morl', 'gaus1', 'gaus8', 'gaus5']]},
         'low_freq': {'hyperopt-dist': hp.choice, 'sampling-scope': [[True, False]]}},
    'fourier_basis':
        {'threshold': {'hyperopt-dist': hp.choice, 'sampling-scope': [list(np.arange(0.75, 0.99, 0.05))]},
         'low_rank': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(2, 30, 3)]]},
         'approximation': {'hyperopt-dist': hp.choice, 'sampling-scope': [['smooth', 'exact']]},
         'output_format': {'hyperopt-dist': hp.choice, 'sampling-scope': [['signal', 'spectrum']]}
         },
    'topological_extractor':
        {'window_size_as_share': {'hyperopt-dist': hp.uniform, 'sampling-scope': [0.05, 0.5]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[1, 2, 3, 4, 5]]},
         'delay': {'hyperopt-dist': hp.choice, 'sampling-scope': [[1, 2, 3]]},
         'filtration_type': {'hyperopt-dist': hp.choice, 'sampling-scope': [['vietoris-rips', 'alpha']]},
         'multivariate_strategy': {'hyperopt-dist': hp.choice, 'sampling-scope': [['independent', 'joint']]},
         },
    'quantile_extractor':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]},
         'add_global_features': {'hyperopt-dist': hp.choice, 'sampling-scope': [[True, False]]}},
    'riemann_extractor':
        {'estimator': {'hyperopt-dist': hp.choice, 'sampling-scope': [['corr',
                                                                       'cov', 'lwf', 'mcd', 'hub']]},
         'tangent_metric': {'hyperopt-dist': hp.choice, 'sampling-scope': [[
             'euclid',
             'logeuclid',
             'riemann'
         ]]},
         'SPD_metric': {'hyperopt-dist': hp.choice, 'sampling-scope': [[
             'euclid',
             'identity',
             'logeuclid', 'riemann']]}},
    'recurrence_extractor':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]},
         'rec_metric': {'hyperopt-dist': hp.choice, 'sampling-scope': [['cosine', 'euclidean']]}
         # 'image_mode': {'hyperopt-dist': hp.choice, 'sampling-scope': [[True, False]]}
         },
    'minirocket_extractor':
        {'num_features': {'hyperopt-dist': hp.choice,
                          'sampling-scope': [[x for x in range(5000, 20000, 1000)]]}},
    'chronos_extractor':
        {'num_features': {'hyperopt-dist': hp.choice,
                          'sampling-scope': [[x for x in range(5000, 20000, 1000)]]}},
    'channel_filtration':
        {'distance': {'hyperopt-dist': hp.choice,
                      'sampling-scope': [['manhattan', 'euclidean', 'chebyshev']]},
         'centroid_metric': {'hyperopt-dist': hp.choice,
                             'sampling-scope': [['manhattan', 'euclidean', 'chebyshev']]},
         'sample_metric': {'hyperopt-dist': hp.choice,
                           'sampling-scope': [['manhattan', 'euclidean', 'chebyshev']]},

         'selection_strategy': {'hyperopt-dist': hp.choice,
                                'sampling-scope': [['sum', 'pairwise']]}
         },
    'patch_tst_model':
        {'patch_len': {'hyperopt-dist': hp.choice, 'sampling-scope': [[8, 12, 16, 20, 24, 32]]},
         'activation': {'hyperopt-dist': hp.choice,
                        'sampling-scope': [
                            ['LeakyReLU', 'ELU', 'SwishBeta', 'ReLU', 'Tanh', 'Softmax', 'SmeLU', 'Mish']]}},
    'tst_model':
        {'activation': {'hyperopt-dist': hp.choice,
                        'sampling-scope': [['GELU', 'ReLU', 'LeakyReLU', 'ELU', 'Tanh']]},
         'model_dim': {'hyperopt-dist': hp.choice, 'sampling-scope': [[64, 128, 256]]},
         'n_layers': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 3, 4, 5]]},
         'number_heads': {'hyperopt-dist': hp.choice, 'sampling-scope': [[4, 8, 16]]},
         'd_ff': {'hyperopt-dist': hp.choice, 'sampling-scope': [[128, 256, 512]]},
         'dropout': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.05, 0.1, 0.2, 0.3]]}},
    'deepar_model':
        {'patch_len': {'hyperopt-dist': hp.choice, 'sampling-scope': [[8, 12, 16, 20, 24, 32]]},
         'cell_type': {'hyperopt-dist': hp.choice, 'sampling-scope': [['GRU', 'LSTM', 'RNN']]},
         'rnn_layers': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 5, 1)]]},
         'hidden_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(10, 50, 5)]]},
         'expected_distribution': {'hyperopt-dist': hp.choice, 'sampling-scope': [['normal', 'cauchy']]},
         'dropout': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.05, 0.1, 0.3, 0.5]]}
         # 'activation': {'hyperopt-dist': hp.choice,
         #                'sampling-scope': [
         #                    ['LeakyReLU', 'SwishBeta', 'ReLU', 'Tanh']]}
         },
    'inception_model':
        {'epochs': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(150, 500, 50)]]},
         'activation': {'hyperopt-dist': hp.choice,
                        'sampling-scope': [
                            ['LeakyReLU', 'SwishBeta', 'Tanh', 'Softmax', 'SmeLU', 'Mish']]}},
    'resnet_model':
        {'epochs': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(150, 500, 50)]]},
         'activation': {'hyperopt-dist': hp.choice,
                        'sampling-scope': [
                            ['LeakyReLU', 'SwishBeta', 'Tanh', 'Softmax', 'SmeLU', 'Mish']]}},
    'xcm_model':
        {'epochs': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(150, 500, 50)]]},
         'activation': {'hyperopt-dist': hp.choice,
                        'sampling-scope': [
                            ['LeakyReLU', 'SwishBeta', 'Tanh', 'Softmax', 'SmeLU', 'Mish']]}},

    'tcn_model':
        {'patch_len': {'hyperopt-dist': hp.choice, 'sampling-scope': [[8, 12, 16, 20, 24, 32]]},
         'activation': {'hyperopt-dist': hp.choice,
                        'sampling-scope': [
                            ['LeakyReLU', 'SwishBeta', 'Tanh', 'Softmax', 'SmeLU', 'Mish', 'ReLU', 'GELU']]},
         'kernel_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 3, 5, 7]]},
         'num_filters': {'hyperopt-dist': hp.choice, 'sampling-scope': [[8, 16, 32, 64]]},
         'num_layers': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 3, 4, 5]]},
         'dilation_base': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 3, 4]]},
         'dropout': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.05, 0.1, 0.2, 0.3]]},
         'weight_norm': {'hyperopt-dist': hp.choice, 'sampling-scope': [[True, False]]}},

    'topo_forecaster':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(8, 48, 4)]]},
         'patch_len': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(4, 24, 2)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[1, 2, 3, 4]]},
         'alpha': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.1, 0.5, 1.0, 2.0, 5.0]]}},
    'hankelisation':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(8, 48, 4)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 6, 1)]]}},
    'svd_decomposition': {},
    'randomized_svd_decomposition':
        {'n_oversamples': {'hyperopt-dist': hp.choice,
                           'sampling-scope': [[3, 5, 8, 10]]}},
    'tensor_decomposition':
        {'unfolding_strategy': {'hyperopt-dist': hp.choice,
                                'sampling-scope': [['channels_last', 'flat_table']]}},
    'explained_variance_truncation':
        {'explained_variance': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.85, 0.9, 0.95, 0.98]]},
         'min_rank': {'hyperopt-dist': hp.choice, 'sampling-scope': [[1, 2, 3]]}},
    'statistical_rank_truncation':
        {'min_rank': {'hyperopt-dist': hp.choice,
                      'sampling-scope': [[1, 2, 3]]}},
    'expert_rank_truncation':
        {'rank': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 4, 6, 8, 12]]},
         'min_rank': {'hyperopt-dist': hp.choice, 'sampling-scope': [[1, 2, 3]]}},
    'lagged_forecaster':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(8, 48, 4)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[1, 2, 3, 4]]},
         'alpha': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.1, 0.5, 1.0, 2.0, 5.0]]}},
    'low_rank_lagged_ridge_forecaster':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(8, 48, 4)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[1, 2, 3, 4]]},
         'alpha': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.1, 0.5, 1.0, 2.0, 5.0]]},
         'explained_variance': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.85, 0.9, 0.95, 0.98]]},
         'decomposition_strategy': {'hyperopt-dist': hp.choice, 'sampling-scope': [['full', 'randomized']]},
         'rank_truncation_policy': {'hyperopt-dist': hp.choice,
                                    'sampling-scope': [['explained_variance', 'statistical']]}},
    'hybrid_ensemble_forecaster':
        {'complex_branch': {'hyperopt-dist': hp.choice, 'sampling-scope': [['okhs', 'havok']]},
         'calibration_horizon': {'hyperopt-dist': hp.choice, 'sampling-scope': [[None, 4, 6, 8]]}},
    'ssa_forecaster':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(8, 48, 4)]]},
         'rank': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 4, 6, 8]]},
         'explained_variance': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.85, 0.9, 0.95, 0.98]]},
         'history_lookback': {'hyperopt-dist': hp.choice, 'sampling-scope': [[20, 30, 40, 60]]},
         'head_policy': {'hyperopt-dist': hp.choice, 'sampling-scope': [['mlp', 'linear']]},
         'head_hidden_dim': {'hyperopt-dist': hp.choice, 'sampling-scope': [[32, 64, 96]]},
         'head_hidden_layers': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 3]]},
         'head_epochs': {'hyperopt-dist': hp.choice, 'sampling-scope': [[60, 120, 180]]},
         'head_learning_rate': {'hyperopt-dist': hp.choice, 'sampling-scope': [[5e-4, 1e-3, 2e-3]]}},
    'mssa_forecaster':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(8, 48, 4)]]},
         'rank': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 4, 6, 8]]},
         'explained_variance': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.85, 0.9, 0.95, 0.98]]},
         'coupled': {'hyperopt-dist': hp.choice, 'sampling-scope': [[True, False]]},
         'head_policy': {'hyperopt-dist': hp.choice, 'sampling-scope': [['mlp', 'linear']]},
         'head_hidden_dim': {'hyperopt-dist': hp.choice, 'sampling-scope': [[32, 64, 96]]},
         'head_hidden_layers': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 3]]},
         'head_epochs': {'hyperopt-dist': hp.choice, 'sampling-scope': [[60, 120, 180]]},
         'head_learning_rate': {'hyperopt-dist': hp.choice, 'sampling-scope': [[5e-4, 1e-3, 2e-3]]}},
    'havok_forecaster':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[8, 12, 16, 20, 24]]},
         'rank': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 4, 6, 8]]},
         'forcing_threshold_scale': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.75, 1.0, 1.25, 1.5]]},
         'forcing_decay': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.7, 0.8, 0.85, 0.9]]},
         'head_policy': {'hyperopt-dist': hp.choice, 'sampling-scope': [['mlp', 'linear']]},
         'head_activation': {'hyperopt-dist': hp.choice, 'sampling-scope': [['relu', 'gelu', 'tanh', 'elu']]},
         'head_depth': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 4, 6, 8]]}},
    'okhs_fdmd_forecaster':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(8, 48, 4)]]},
         'n_modes': {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 4, 6, 8]]},
         'q': {'hyperopt-dist': hp.choice, 'sampling-scope': [[0.55, 0.7, 0.85]]},
         'trajectory_sampling_policy': {'hyperopt-dist': hp.choice, 'sampling-scope': [['dense', 'adaptive_stride']]},
         'trajectory_rank_policy': {'hyperopt-dist': hp.choice, 'sampling-scope': [['explained_dispersion', 'none']]},
         'trajectory_representation_policy': {'hyperopt-dist': hp.choice,
                                              'sampling-scope': [['projected', 'reconstructed']]}},
    'industrial_stat_clf':
        {'channel_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['logit', 'xgboost', 'rf',
                                                                           # 'inception_model','resnet_model'
                                                                           ]]},
         'transformation_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['quantile_extractor']]}
         },
    'industrial_freq_clf':
        {'channel_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['logit', 'xgboost', 'rf',
                                                                           # 'inception_model','resnet_model'
                                                                           ]]},
         'transformation_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['fourier_basis',
                                                                                  'wavelet_basis',
                                                                                  'eigen_basis'
                                                                                  ]]}
         },
    'industrial_manifold_clf':
        {'channel_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['logit', 'xgboost', 'rf',
                                                                           # 'inception_model','resnet_model'
                                                                           ]]},
         'transformation_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['recurrence_extractor',
                                                                                  'riemann_extractor']]}
         },
    'industrial_stat_reg':
        {'channel_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['treg', 'ridge', 'xgbreg',
                                                                           # 'inception_model','resnet_model'
                                                                           ]]},
         'transformation_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['quantile_extractor']]}
         },
    'industrial_freq_reg':
        {'channel_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['treg', 'ridge', 'xgbreg',
                                                                           # 'inception_model','resnet_model'
                                                                           ]]},
         'transformation_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['fourier_basis',
                                                                                  'wavelet_basis',
                                                                                  'eigen_basis'
                                                                                  ]]}
         },
    'industrial_manifold_reg':
        {'channel_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['treg', 'ridge', 'xgbreg',
                                                                           # 'inception_model','resnet_model'
                                                                           ]]},
         'transformation_model': {'hyperopt-dist': hp.choice, 'sampling-scope': [['recurrence_extractor',
                                                                                  'riemann_extractor']]}
         },
    'nbeats_model':
        {"n_stacks": {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(10, 50, 10)]]},
         "n_trend_blocks": {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 5, 1)]]},
         "n_seasonality_blocks": {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 4, 1)]]},
         "n_of_harmonics": {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 3, 1)]]},
         "layers": {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 4, 6]]},
         "degree_of_polynomial": {'hyperopt-dist': hp.choice, 'sampling-scope': [[2, 4, 6, 8]]}},
    'bagging': {'method':
                {'hyperopt-dist': hp.choice, 'sampling-scope': [['max', 'min', 'mean', 'median']]}},
    'stat_detector':
        {'anomaly_thr': {'hyperopt-dist': hp.choice, 'sampling-scope': [list(np.arange(0.75, 0.99, 0.05))]},
         'window_length': {'hyperopt-dist': hp.choice,
                           'sampling-scope': [list(np.arange(10, 35, 5))]}},
    'arima_detector':
        {'anomaly_thr': {'hyperopt-dist': hp.choice, 'sampling-scope': [list(np.arange(0.75, 0.99, 0.05))]},
         'window_length': {'hyperopt-dist': hp.choice,
                           'sampling-scope': [list(np.arange(10, 35, 5))]}},
    'iforest_detector':
        {'anomaly_thr': {'hyperopt-dist': hp.choice, 'sampling-scope': [list(np.arange(0.05, 0.5, 0.05))]},
         'window_length': {'hyperopt-dist': hp.choice,
                           'sampling-scope': [list(np.arange(10, 35, 5))]}},
    'conv_ae_detector':
        {'anomaly_thr': {'hyperopt-dist': hp.choice, 'sampling-scope': [list(np.arange(0.75, 0.99, 0.05))]},
         'window_length': {'hyperopt-dist': hp.choice,
                           'sampling-scope': [list(np.arange(10, 35, 5))]}},
    'lstm_ae_detector':
        {'anomaly_thr': {'hyperopt-dist': hp.choice, 'sampling-scope': [list(np.arange(0.75, 0.99, 0.05))]},
         'window_length': {'hyperopt-dist': hp.choice,
                           'sampling-scope': [list(np.arange(10, 35, 5))]}},
    'pdl_clf': {},
    'pdl_reg': {}
}

pdl_base_model = {'pdl_clf': 'rf',
                  'pdl_reg': 'treg'}

for _forecasting_alias, _canonical_name in FORECASTING_MODEL_ALIASES.items():
    if _canonical_name in industrial_search_space and _forecasting_alias not in industrial_search_space:
        industrial_search_space[_forecasting_alias] = industrial_search_space[_canonical_name]


def get_industrial_search_space(config=None):
    """Return an independent FEDOT and Industrial parameter space mapping."""
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace

    parameters = deepcopy(PipelineSearchSpace().get_parameters_dict())
    industrial_parameters = deepcopy(industrial_search_space)
    for operation, base_operation in pdl_base_model.items():
        if base_operation in parameters:
            industrial_parameters[operation] = deepcopy(parameters[base_operation])
    parameters.update(industrial_parameters)

    custom_search_space = getattr(config, 'custom_search_space', None)
    replace_default = getattr(config, 'replace_default_search_space', False)
    if custom_search_space is not None:
        for operation, operation_parameters in custom_search_space.items():
            if replace_default or operation not in parameters:
                parameters[operation] = deepcopy(operation_parameters)
            else:
                parameters[operation].update(deepcopy(operation_parameters))
    return parameters


def build_industrial_pipeline_search_space(config=None):
    """Build an explicit FEDOT search-space object without class monkeypatches."""
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace

    return PipelineSearchSpace(
        custom_search_space=get_industrial_search_space(config),
        replace_default_search_space=True,
    )

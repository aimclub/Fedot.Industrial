from functools import partial

from hyperopt import hp

from fedot_ind.core.repository.constanst_repository import DISTANCE_METRICS

NESTED_PARAMS_LABEL = 'nested_label'

industrial_search_space = {
    'eigen_basis':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]},
         'rank_regularization': {'hyperopt-dist': hp.choice, 'sampling-scope': [
             ['hard_thresholding', 'explained_dispersion']]}},
    'wavelet_basis':
        {'n_components': {'hyperopt-dist': hp.uniformint, 'sampling-scope': [2, 10]},
         'wavelet': {'hyperopt-dist': hp.choice,
                     'sampling-scope': [['mexh', 'morl', 'db5', 'sym5']]}},
    'fourier_basis':
        {'threshold': {'hyperopt-dist': hp.uniformint,
                       'sampling-scope': [10000, 50000]}},
    'topological_extractor':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]}},
    'quantile_extractor':
        {'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]},
         'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(0, 50, 3)]]}},
    'riemann_extractor':
        {'estimator': {'hyperopt-dist': hp.choice, 'sampling-scope': [['corr',
                                                                       'cov', 'lwf', 'mcd', 'hub']]},
         'tangent_metric': {'hyperopt-dist': hp.choice, 'sampling-scope': [[
             'euclid',
             'logeuclid',
             'riemann'
         ]]},
         'SPD_metric': {'hyperopt-dist': hp.choice, 'sampling-scope': [[
             # 'ale',
             # 'alm',
             'euclid',
             'identity',
             'logeuclid', 'riemann']]}},
    'recurrence_extractor':
        {'window_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(5, 50, 5)]]},
         'stride': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(1, 10, 1)]]},
         # 'rec_metric': (hp.choice, [['chebyshev', 'cosine', 'euclidean', 'mahalanobis']])
         },
    'signal_extractor':
        {'n_components': {'hyperopt-dist': hp.uniformint, 'sampling-scope': [2, 10]},
         'wavelet': {'hyperopt-dist': hp.choice,
                     'sampling-scope': [['mexh', 'morl', 'db5', 'sym5']]}},
    'channel_filtration':
        {'distance': {'hyperopt-dist': hp.choice,
                      'sampling-scope': [['manhattan', 'euclidean', 'chebyshev']]},
         'centroid_metric': {'hyperopt-dist': hp.choice,
                             'sampling-scope': [['manhattan', 'euclidean', 'chebyshev']]},
         'sample_metric': {'hyperopt-dist': hp.choice,
                           'sampling-scope': [list(DISTANCE_METRICS.keys())]},

         'selection_strategy': {'hyperopt-dist': hp.choice,
                                'sampling-scope': [['sum', 'pairwise']]}
         },
    'minirocket_extractor':
        {'num_features': {'hyperopt-dist': hp.choice,
                          'sampling-scope': [[x for x in range(5000, 20000, 1000)]]}},
    'chronos_extractor':
        {'num_features': {'hyperopt-dist': hp.choice,
                          'sampling-scope': [[x for x in range(5000, 20000, 1000)]]}},
    'patch_tst_model':
        {'epochs': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(10, 100, 10)]]},
         'batch_size': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(8, 64, 6)]]},
         'activation': {'hyperopt-dist': hp.choice,
                        'sampling-scope': [
                            ['LeakyReLU', 'ELU', 'SwishBeta', 'ReLU', 'Tanh', 'Softmax', 'SmeLU', 'Mish']]}},
    'omniscale_model':
        {'epochs': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(150, 500, 50)]]},
         'activation': {'hyperopt-dist': hp.choice,
                        'sampling-scope': [
                            ['LeakyReLU', 'ELU', 'SwishBeta', 'Tanh', 'Softmax', 'SmeLU', 'Mish']]}},
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
    'ssa_forecaster':
        {'window_size_method': {'hyperopt-dist': hp.choice,
                                'sampling-scope': [['hac', 'dff']]},
         'history_lookback': {'hyperopt-dist': hp.choice, 'sampling-scope': [[x for x in range(30, 300, 30)]]}, }
}


def get_industrial_search_space(self):
    parameters_per_operation = {'kmeans': {
        'n_clusters': {
            'hyperopt-dist': hp.uniformint,
            'sampling-scope': [2, 7],
            'type': 'discrete'}
    },
        'adareg': {
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'},
            'loss': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["linear", "square", "exponential"]],
                'type': 'categorical'}
        },
        'gbr': {
            'loss': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["ls", "lad", "huber", "quantile"]],
                'type': 'categorical'},
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'},
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 11],
                'type': 'discrete'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 21],
                'type': 'discrete'},
            'subsample': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'},
            'max_features': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'},
            'alpha': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.75, 0.99],
                'type': 'continuous'}
        },
        'logit': {
            'C': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-2, 10.0],
                'type': 'continuous'},

            'penalty': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['l1', 'l2']],
                'type': 'categorical'},

            'solver': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['liblinear']],
                'type': 'categorical'}
        },
        'rf': {
            'criterion': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["gini", "entropy"]],
                'type': 'categorical'},
            'max_features': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 10],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 15],
                'type': 'discrete'},
            'bootstrap': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[True, False]],
                'type': 'categorical'}
        },
        'ridge': {
            'alpha': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.01, 10.0],
                'type': 'continuous'}
        },
        'lasso': {
            'alpha': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.01, 10.0],
                'type': 'continuous'}
        },
        'rfr': {
            'max_features': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 15],
                'type': 'discrete'},
            'bootstrap': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[True, False]],
                'type': 'categorical'}
        },
        'xgbreg': {
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 11],
                'type': 'discrete'},
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'},
            'subsample': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'},
            'min_child_weight': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 21],
                'type': 'discrete'},
        },
        'xgboost': {
            'n_estimators': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [100, 3000],
                'type': 'discrete'},
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [3, 10],
                'type': 'discrete'},
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'},
            'subsample': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 0.99],
                'type': 'continuous'},
            'min_weight_fraction_leaf': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.0, 0.5],
                'type': 'continuous'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.0, 1],
                'type': 'continuous'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.0, 1.0],
                'type': 'continuous'}
        },
        'svr': {
            'C': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-4, 25.0],
                'type': 'continuous'},
            'epsilon': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-4, 1],
                'type': 'continuous'},
            'tol': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-5, 1e-1],
                'type': 'continuous'},
            'loss': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["epsilon_insensitive", "squared_epsilon_insensitive"]],
                'type': 'categorical'}
        },
        'dtreg': {
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 11],
                'type': 'discrete'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 21],
                'type': 'discrete'}
        },
        'treg': {
            'max_features': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 21],
                'type': 'discrete'},
            'bootstrap': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[True, False]],
                'type': 'categorical'}
        },
        'dt': {
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 11],
                'type': 'discrete'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 21],
                'type': 'discrete'}
        },
        'knnreg': {
            'n_neighbors': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 50],
                'type': 'discrete'},
            'weights': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["uniform", "distance"]],
                'type': 'categorical'},
            'p': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[1, 2]],
                'type': 'categorical'}
        },
        'knn': {
            'n_neighbors': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 50],
                'type': 'discrete'},
            'weights': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["uniform", "distance"]],
                'type': 'categorical'},
            'p': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[1, 2]],
                'type': 'categorical'}
        },
        'arima': {
            'p': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 7],
                'type': 'discrete'},
            'd': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [0, 2],
                'type': 'discrete'},
            'q': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 5],
                'type': 'discrete'}
        },
        'stl_arima': {
            'p': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 7],
                'type': 'discrete'},
            'd': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [0, 2],
                'type': 'discrete'},
            'q': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 5],
                'type': 'discrete'},
            'period': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 365],
                'type': 'discrete'}
        },
        'mlp': {
            'hidden_layer_sizes': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[(256, 128, 64, 32), (1028, 512, 64,)]],
                'type': 'categorical'},
            'activation': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['logistic', 'tanh', 'relu']],
                'type': 'categorical'},
            'max_iter': {'hyperopt-dist': hp.uniformint,
                         'sampling-scope': [1000, 2000],
                         'type': 'discrete'},
            'learning_rate': {'hyperopt-dist': hp.choice,
                              'sampling-scope': [['constant', 'adaptive']],
                              'type': 'categorical'}
        },
        'ar': {
            'lag_1': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [2, 200],
                'type': 'continuous'},
            'lag_2': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [2, 800],
                'type': 'continuous'}
            ,
            'trend': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['n', 'c', 't', 'ct']],
                'type': 'categorical'}
        },
        'ets': {
            'error': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["add", "mul"]],
                'type': 'categorical'},
            'trend': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[None, "add", "mul"]],
                'type': 'categorical'},
            'seasonal': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[None, "add", "mul"]],
                'type': 'categorical'},
            'damped_trend': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[True, False]],
                'type': 'categorical'},
            'seasonal_periods': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1, 100],
                'type': 'continuous'}
        },
        'glm': {
            NESTED_PARAMS_LABEL: {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[
                    {
                        'family': 'gaussian',
                        'link': hp.choice('link_gaussian', ['identity',
                                                            'inverse_power',
                                                            'log'])
                    },
                    {
                        'family': 'gamma',
                        'link': hp.choice('link_gamma', ['identity',
                                                         'inverse_power',
                                                         'log'])
                    },
                    {
                        'family': 'inverse_gaussian',
                        'link': hp.choice('link_inv_gaussian', ['identity',
                                                                'inverse_power'])
                    }

                ]],
                'type': 'categorical'}
        },
        'cgru': {
            'hidden_size': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [20, 200],
                'type': 'continuous'},
            'learning_rate': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.0005, 0.005],
                'type': 'continuous'},
            'cnn1_kernel_size': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [3, 8],
                'type': 'discrete'},
            'cnn1_output_size': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[8, 16, 32, 64]],
                'type': 'categorical'},
            'cnn2_kernel_size': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [3, 8],
                'type': 'discrete'},
            'cnn2_output_size': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[8, 16, 32, 64]],
                'type': 'categorical'},
            'batch_size': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[64, 128]],
                'type': 'categorical'},
            'num_epochs': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[10, 20, 50, 100]],
                'type': 'categorical'},
            'optimizer': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['adamw', 'sgd']],
                'type': 'categorical'},
            'loss': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['mae', 'mse']],
                'type': 'categorical'},
        },
        'topological_extractor': {
            'window_size_as_share': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.1, 0.9],
                'type': 'continuous'
            },
            'max_homology_dimension': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 3],
                'type': 'discrete'
            },
            'metric': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['euclidean', 'manhattan', 'cosine']],
                'type': 'categorical'}
        },
        'pca': {
            'n_components': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.1, 0.99],
                'type': 'continuous'}
        },
        'kernel_pca': {
            'n_components': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 20],
                'type': 'discrete'},
            'kernel': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed']],
                'type': 'categorical'}
        },
        'lagged': {
            'window_size': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [5, 500],
                'type': 'discrete'}
        },
        'sparse_lagged': {
            'window_size': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [5, 500],
                'type': 'discrete'},
            'n_components': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0, 0.5],
                'type': 'continuous'},
            'use_svd': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[True, False]],
                'type': 'categorical'}
        },
        'smoothing': {
            'window_size': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 20],
                'type': 'discrete'}
        },
        'gaussian_filter': {
            'sigma': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1, 5],
                'type': 'continuous'}
        },
        'diff_filter': {
            'poly_degree': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 5],
                'type': 'discrete'},
            'order': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1, 3],
                'type': 'continuous'},
            'window_size': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [3, 20],
                'type': 'continuous'}
        },
        'cut': {
            'cut_part': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0, 0.9],
                'type': 'continuous'}
        },
        'lgbm': {
            'class_weight': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[None, 'balanced']],
                'type': 'categorical'},
            'num_leaves': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 256],
                'type': 'discrete'},
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [0.01, 0.2],
                'type': 'continuous'},
            'colsample_bytree': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.4, 1],
                'type': 'continuous'},
            'subsample': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.4, 1],
                'type': 'continuous'},
            'reg_alpha': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-8, 10],
                'type': 'continuous'},
            'reg_lambda': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-8, 10],
                'type': 'continuous'}
        },
        'lgbmreg': {
            'num_leaves': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 256],
                'type': 'discrete'},
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [0.01, 0.2],
                'type': 'continuous'},
            'colsample_bytree': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.4, 1],
                'type': 'continuous'},
            'subsample': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.4, 1],
                'type': 'continuous'},
            'reg_alpha': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-8, 10],
                'type': 'continuous'},
            'reg_lambda': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-8, 10],
                'type': 'continuous'}
        },
        'catboost': {
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 11],
                'type': 'discrete'},
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [0.01, 0.2],
                'type': 'continuous'},
            'min_data_in_leaf': {
                'hyperopt-dist': partial(hp.qloguniform, q=1),
                'sampling-scope': [0, 6],
                'type': 'discrete'},
            'border_count': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 255],
                'type': 'discrete'},
            'l2_leaf_reg': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-8, 10],
                'type': 'continuous'}
        },
        'catboostreg': {
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 11],
                'type': 'discrete'},
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [0.01, 0.2],
                'type': 'continuous'},
            'min_data_in_leaf': {
                'hyperopt-dist': partial(hp.qloguniform, q=1),
                'sampling-scope': [0, 6],
                'type': 'discrete'},
            'border_count': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 255],
                'type': 'discrete'},
            'l2_leaf_reg': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-8, 10],
                'type': 'continuous'}
        },
        'resample': {
            'balance': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['expand_minority', 'reduce_majority']],
                'type': 'categorical'},
            'replace': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[True, False]],
                'type': 'categorical'},
            'balance_ratio': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.3, 1],
                'type': 'continuous'}
        },
        'lda': {
            'solver': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['svd', 'lsqr', 'eigen']],
                'type': 'categorical'},
            'shrinkage': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.1, 0.9],
                'type': 'continuous'}
        },
        'ts_naive_average': {
            'part_for_averaging': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.1, 1],
                'type': 'continuous'}
        },
        'locf': {
            'part_for_repeat': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.01, 0.5],
                'type': 'continuous'}
        },
        'word2vec_pretrained': {
            'model_name': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [['glove-twitter-25', 'glove-twitter-50',
                                    'glove-wiki-gigaword-100', 'word2vec-ruscorpora-300']],
                'type': 'categorical'}
        },
        'tfidf': {
            'ngram_range': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[(1, 1), (1, 2), (1, 3)]],
                'type': 'categorical'},
            'min_df': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.0001, 0.1],
                'type': 'continuous'},
            'max_df': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.9, 0.99],
                'type': 'continuous'}
        },
    }
    for key in industrial_search_space:
        parameters_per_operation[key] = industrial_search_space[key]

    if self.custom_search_space is not None:
        for operation in self.custom_search_space.keys():
            if self.replace_default_search_space:
                parameters_per_operation[operation] = self.custom_search_space[operation]
            else:
                for key, value in self.custom_search_space[operation].items():
                    parameters_per_operation[operation][key] = value

    return parameters_per_operation

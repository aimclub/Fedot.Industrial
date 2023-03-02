from hyperopt import hp

industrial_search_space = {
    'data_driven_basic':
        {'n_components': (hp.uniformint, [2, 10]),
         'window_length': (hp.uniformint, [10, 50])},
    'quantile_extractor':
        {'window_size': (hp.uniformint, [1, 50]),
         'win_mode': (hp.choice, [True, False])},
    'recurrence_extractor':
        {'window_size': (hp.uniformint, [1, 50]),
         'win_mode': (hp.choice, [True, False]),
         'min_signal_ratio': (hp.uniform, [0, 0.5]),
         'max_signal_ratio': (hp.uniform, [0.5, 1]),
         'rec_metric': (hp.choice, ['chebyshev', 'cosine', 'euclidean' 'mahalanobis'])},
    'signal_extractor':
        {'wavelet': (hp.choice, ['db5', 'sym5', 'coif5', 'bior2.4'])}
}

import numpy as np


def quantile(column, q: str):
    return np.quantile(a=column, q=q)


stat_methods_default = {
    'mean_': np.mean,
    'median_': np.median,
    'std_': np.std,
    'var_': np.var,
    'q5_': quantile,
    'q25_': quantile,
    'q75_': quantile,
    'q95_': quantile,
}

stat_methods_full = {
    'mean_': np.mean,
    'median_': np.median,
    'lambda_less_zero': lambda x: x < 0.01,
    'std_': np.std,
    'var_': np.var,
    'max': np.max,
    'min': np.min,
    'q5_': quantile,
    'q25_': quantile,
    'q75_': quantile,
    'q95_': quantile,
    'sum_': np.sum
}

hyperparam_dict = {'statistical_methods': stat_methods_default,
                   'statistical_methods_extra': stat_methods_full}


def ParamSelector(param_name):
    return hyperparam_dict[param_name]

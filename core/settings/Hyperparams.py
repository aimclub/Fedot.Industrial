import numpy as np


def quantile(column, q: str):
    return np.quantile(a=column, q=q)

stat_methods = {
    'mean_': np.mean,
    'median_': np.median,
    'std_': np.std,
    'var_': np.var,
    # 'max': np.max,
    # 'min': np.min,
    'q5_': quantile,
    'q25_': quantile,
    'q75_': quantile,
    'q95_': quantile,
    # 'sum_': np.sum
}


hyperparam_dict = {'statistical_methods': stat_methods}


def ParamSelector(param_name):
    return hyperparam_dict[param_name]

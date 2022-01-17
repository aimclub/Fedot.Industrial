import numpy as np

radius_near_bus_stop = 0.0003
stat_methods = {'mean_': np.mean,
                'median_': np.median,
                'std_': np.std,
                'var_': np.var,
                'max': np.max,
                'min': np.min,
                'q5_': np.quantile(0.05),
                'q25_': np.quantile(0.25),
                'q75_': np.quantile(0.75),
                'q95_': np.quantile(0.95),
                'sum_': np.sum
                }


target_dict_bus = {'OriginalDriver': 0, 'Bus': 1}

target_dict_pas = {'OriginalDriver': 0, 'Bus': 1, 'Passanger': 2}

target_dict_moto = {'OriginalDriver': 0, 'Moto': 1}

OS_DICT = {'iOS': 0, 'Android': 1}

hyperparam_dict = {'bus_station_radius_to_detect': radius_near_bus_stop,
                   'statistical_methods': stat_methods}



def ParamSelector(param_name):
    return hyperparam_dict[param_name]

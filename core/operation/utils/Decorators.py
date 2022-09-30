import os
import timeit

import pandas as pd

from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import PROJECT_PATH


def exception_decorator(exception_return='Problem'):
    def decorate(function):
        def exception_wrapper(*args, **kwargs):
            try:
                function(*args, **kwargs)
            except:
                return print(exception_return)

        return exception_wrapper

    return decorate


def type_check_decorator(object_type: type, types_list: tuple):
    def type_check_wrapper(function_to_decorate):
        def wrapper(*args, **kwargs):
            if not isinstance(object_type, types_list):
                raise TypeError(f"Unsupported object type. Try one of {str(types_list)}.")
            else:
                function_to_decorate(*args, **kwargs)

            return wrapper

        return type_check_wrapper


# def cache_it(func):
#     def wrapper(runner, ts_data, dataset_name):
#         r = str(type(runner)).split("'")[-2].split('.')[-1]
#         file = os.path.join(PROJECT_PATH, 'cached_features', f'{r}_{dataset_name}.hdf5')
#         if not os.path.isfile(file):
#             features = func(runner, ts_data, dataset_name)
#             features.to_hdf(file, 'features')
#         else:
#             features = pd.read_hdf(file)
#         return features
#
#     return wrapper

def cache_it(switch='off'):
    def inner(func):
        if switch == 'on':
            def wrapper(runner, ts_data, dataset_name):

                r = str(type(runner)).split("'")[-2].split('.')[-1]
                file = os.path.join(PROJECT_PATH, 'cached_features', f'{r}_{dataset_name}.h5')
                if not os.path.isfile(file):
                    features = func(runner, ts_data, dataset_name)
                    features.to_hdf(file, 'features')
                else:
                    features = pd.read_hdf(file)
                return features

            return wrapper
        else:
            return func

    return inner  # this is the fun_obj mentioned in the above content


def time_it(func):
    def wrapper(*args, **kwargs):
        logger = Logger().get_logger()
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        logger.info(f'Time spent on feature generation - {round((end - start), 2)} sec')
        return result

    return wrapper
import logging
import timeit

import pandas as pd


def exception_decorator(exception_return='Problem'):
    def decorate(function):
        def exception_wrapper(*args, **kwargs):
            try:
                function(*args, **kwargs)
            except:
                return print(exception_return)

        return exception_wrapper

    return decorate


def input_data_decorator(func):
    def wrapper(*args, **kwargs):
        data_type = type(kwargs['modelling_results'])
        if data_type != dict:
            if data_type == list:
                modelling_result = [pd.read_csv(element, index_col=0) for element in kwargs['modelling_results']]
                kwargs['modelling_results'] = {str(idx): element for idx, element in enumerate(modelling_result)}
            else:
                kwargs['modelling_results'] = pd.DataFrame(kwargs['modelling_results'])
        result = func(*args, **kwargs)
        return result

    return wrapper


def type_check_decorator(object_type: type, types_list: tuple):
    def type_check_wrapper(function_to_decorate):
        def wrapper(*args, **kwargs):
            if not isinstance(object_type, types_list):
                raise TypeError(f"Unsupported object type. Try one of {str(types_list)}.")
            else:
                function_to_decorate(*args, **kwargs)
            return wrapper

        return type_check_wrapper


def time_it(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('timer')
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        logger.info(f'Time spent - {round((end - start), 2)} sec')
        return result

    return wrapper


def dataframe_adapter(func):
    def wrapper(*args, **kwargs):
        if not isinstance(kwargs['ts_frame'], pd.DataFrame):
            if isinstance(kwargs['ts_frame'], list):
                kwargs['ts_frame'] = pd.DataFrame(kwargs['ts_frame'])
            else:
                raise TypeError(f'Unsupported input type for {args[0].__class__.__name__}')
        return func(*args, **kwargs)
    return wrapper


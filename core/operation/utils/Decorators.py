import timeit

from core.operation.utils.LoggerSingleton import Logger


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


def time_it(func):
    def wrapper(*args, **kwargs):
        logger = Logger().get_logger()
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        logger.info(f'Time spent on feature generation - {round((end - start), 2)} sec')
        return result
    return wrapper

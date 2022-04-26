class DecoratorObject:
    def __init__(self, deco_type: str):
        self.deco_type = deco_type

    def __call__(self, function):
        def wrapper():
            value = function()
            return value
        return wrapper


def exception_decorator(function_to_decorate, exception_return=None):
    def exception_wrapper():
        try:
            function_to_decorate()
        except Exception:
            return exception_return
        return exception_wrapper


def logger_decorator(function_to_decorate):
    def logger_wrapper():
        function_to_decorate()
        return logger_wrapper


def type_check_decorator(object_type: type, types_list: tuple):
    def type_check_wrapper(function_to_decorate):
        def wrapper(*args, **kwargs):
            if not isinstance(object_type, types_list):
                raise TypeError(f"Unsupported object type. Try one of {str(types_list)}.")
            else:
                function_to_decorate(*args, **kwargs)

            return wrapper

        return type_check_wrapper

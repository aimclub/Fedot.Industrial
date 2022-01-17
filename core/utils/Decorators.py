def exception_decorator(function_to_decorate):
    def exception_wrapper():
        try:
            function_to_decorate()
        except Exception:
            return None

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

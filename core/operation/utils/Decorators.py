class DecoratorObject:
    def __init__(self,
                 deco_type: str,
                 exception_return=None):
        self.deco_type = deco_type
        self.exception_return = exception_return

    def __call__(self, function, *args):
        def exception_wrapper(function):
            def applicator(*args):
                try:
                    function(*args)
                except Exception as error:
                    return self.exception_return
            return applicator

        def reshape_wrapper():
            ...

        def logger_wrapper():
            function()

        if self.deco_type == 'exception':
            return exception_wrapper
        elif self.deco_type == 'reshape':
            return reshape_wrapper
        elif self.deco_type == 'logger':
            return logger_wrapper


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

import logging


class SingletonMetaLogger(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Logger(object, metaclass=SingletonMetaLogger):
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    def __init__(self):
        self._logger = logging.getLogger('Experiment logger')
        self._logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(self.FORMAT)
        ch.setFormatter(formatter)

        self._logger.addHandler(ch)

    def get_logger(self):
        return self._logger

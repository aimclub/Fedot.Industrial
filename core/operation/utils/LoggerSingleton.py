import datetime
import logging
import os.path

from core.operation.utils.utils import PROJECT_PATH


class SingletonMetaLogger(type):
    """
    Singleton metaclass for Logger
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaLogger, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(object, metaclass=SingletonMetaLogger):
    """
    Class for implementing singleton Logger
    """
    _logger = None

    def __init__(self):
        self._logger = logging.getLogger('FEDOT-TSC')
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
        now = datetime.datetime.now()
        dirname = os.path.join(PROJECT_PATH, 'log')

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        file_handler = logging.FileHandler(dirname + "/log_" + now.strftime("%Y-%m-%d-%H:%M") + ".log",
                                           delay=True,
                                           mode='w')

        stream_handler = logging.StreamHandler()

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(stream_handler)

    def get_logger(self):
        return self._logger


# class CustomFormatter(logging.Formatter):
#
#     grey = "\x1b[38;20m"
#     yellow = "\x1b"
#     red = "\x1b[31;20m"
#     bold_red = "\x1b[31;1m"
#     reset = "\x1b[0m"
#     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#
#     FORMATS = {
#         logging.DEBUG: grey + format + reset,
#         logging.INFO: yellow + format,
#         logging.WARNING: red + format + reset,
#         logging.ERROR: red + format + reset,
#         logging.CRITICAL: bold_red + format + reset
#     }
#
#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)
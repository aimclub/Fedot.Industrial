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

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # formatter = CustomFormatter()
        now = datetime.datetime.now()
        dirname = os.path.join(PROJECT_PATH, 'log')

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fileHandler = logging.FileHandler(dirname + "/log_" + now.strftime("%Y-%m-%d")+".log")

        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

    def get_logger(self):
        return self._logger

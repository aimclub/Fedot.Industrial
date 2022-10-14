import logging
import os.path
from datetime import date, datetime

from core.operation.utils.utils import project_path

MSG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DT_FORMAT = '%H:%M:%S'
DATE_NOW = date.today()
TIME_NOW = datetime.now().strftime("%H-%M")
LOG_PATH = os.path.join(project_path(), 'log', f'Experiment-log-{DATE_NOW}_{TIME_NOW}.log')


class SingletonMetaLogger(type):
    """
    Singleton metaclass for Logger
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Logger(object, metaclass=SingletonMetaLogger):
    """
    Class for implementing singleton Logger
    """

    def __init__(self):
        logging.basicConfig(filename=LOG_PATH)
        self._logger = logging.getLogger('FEDOT-TSC')
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(MSG_FORMAT, DT_FORMAT)
        ch.setFormatter(formatter)

        self._logger.addHandler(ch)

    def get_logger(self):
        return self._logger

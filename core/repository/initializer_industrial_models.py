from pathlib import Path

from fedot.core.repository.operation_types_repository import OperationTypesRepository

# TODO VERY VERY BAD
from core.architecture.utils.utils import PROJECT_PATH


def initialize_industrial_models():
    OperationTypesRepository.__repository_dict__.update({'data_operation':
                                                             {'file': Path(PROJECT_PATH, 'core', 'repository', 'data',
                                                                           'data_operation_repository.json'),
                                                              'initialized_repo': None,
                                                              'default_tags': []}})


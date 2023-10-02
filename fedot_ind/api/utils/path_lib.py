import os
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent.parent.parent)

# Default parameters of feature generators
PATH_TO_DEFAULT_PARAMS = os.path.join(PROJECT_PATH, 'fedot_ind/core/repository/data/default_operation_params.json')

# For results collection
DS_INFO_PATH = os.path.join(PROJECT_PATH, 'core', 'architecture', 'postprocessing', 'ucr_datasets.json')


def default_path_to_save_results() -> str:
    path = PROJECT_PATH
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path

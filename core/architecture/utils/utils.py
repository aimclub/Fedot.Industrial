import os
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent.parent.parent)


def default_path_to_save_results() -> str:
    path = PROJECT_PATH
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path

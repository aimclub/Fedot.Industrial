import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PROJECT_PATH = str(Path(__file__).parent.parent.parent.parent)

def path_to_save_results() -> str:
    path = PROJECT_PATH
    save_path = os.path.join(path, 'results_of_experiments')
    return save_path

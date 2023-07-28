import os
from typing import Optional, Tuple

import numpy as np
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.api.utils.path_lib import default_path_to_save_results
from fedot_ind.core.models.nn.inception import InceptionTimeNetwork

TSCCLF_MODEL = {
    'inception_time': InceptionTimeNetwork
}


class TimeSeriesClassifierNN(TimeSeriesClassifier):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.device = torch.device('cuda' if params.get('gpu', False) else 'cpu')
        self.model = TSCCLF_MODEL[params.get('model', 'inception_time')].network_architecture
        self.num_epochs = params.get('num_epochs', 10)

    def _init_model_param(self, target: np.ndarray) -> Tuple[int, np.ndarray]:
        self.model_hyperparams['models_saving_path'] = os.path.join(default_path_to_save_results(), 'TSCNN',
                                                                    '../../models')
        self.model_hyperparams['summary_path'] = os.path.join(default_path_to_save_results(), 'TSCNN',
                                                              'runs')
        self.model_hyperparams['num_classes'] = np.unique(target).shape[0]

        if target.min() != 0:
            target = target - 1

        return self.num_epochs, target
import logging
from copy import deepcopy

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.ensemble.kernel_ensemble import KernelEnsembler
from fedot_ind.core.ensemble.random_automl_forest import RAFensembler
from fedot_ind.core.repository.constanst_repository import BATCH_SIZE_FOR_FEDOT_WORKER, FEDOT_WORKER_NUM, \
    FEDOT_WORKER_TIMEOUT_PARTITION, FEDOT_TUNING_METRICS, FEDOT_TUNER_STRATEGY

import numpy as np

from fedot_ind.core.repository.industrial_implementations.abstract import build_tuner
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


class IndustrialStrategy:
    def __init__(self, industrial_strategy_params,
                 industrial_strategy,
                 api_config,
                 logger=None
                 ):
        self.industrial_strategy_params = industrial_strategy_params
        self.industrial_strategy = industrial_strategy
        self.industrial_strategy_fit = {'federated_automl': self._federated_strategy,
                                        'kernel_automl': self._kernel_strategy}
        self.industrial_strategy_predict = {'federated_automl': self._federated_predict,
                                            'kernel_automl': self._kernel_predict}
        self.config_dict = api_config
        self.logger = logger
        self.repo = IndustrialModels().setup_repository()

    def fit(self, input_data):
        self.industrial_strategy_fit[self.industrial_strategy](input_data)
        return self.solver

    def predict(self, input_data, predict_mode):
        return self.industrial_strategy_predict[self.industrial_strategy](input_data, predict_mode)

    def _federated_strategy(self, input_data):
        if input_data.features.shape[0] > BATCH_SIZE_FOR_FEDOT_WORKER:
            self.logger.info('RAF algorithm was applied')
            batch_size = round(input_data.features.shape[0] / self.RAF_workers if self.RAF_workers
                                                                                  is not None else FEDOT_WORKER_NUM)
            batch_timeout = round(self.config_dict['timeout'] / FEDOT_WORKER_TIMEOUT_PARTITION)
            self.config_dict['timeout'] = batch_timeout
            self.logger.info(f'Batch_size - {batch_size}. Number of batches - {self.RAF_workers}')
            self.solver = RAFensembler(composing_params=self.config_dict, n_splits=self.RAF_workers,
                                       batch_size=batch_size)
            self.logger.info(f'Number of AutoMl models in ensemble - {self.solver.n_splits}')

    def _finetune_loop(self,
                       kernel_ensemble: dict,
                       kernel_data: dict,
                       tuning_params: dict = {}):
        tuned_kernels = {}
        tuning_params['metric'] = FEDOT_TUNING_METRICS[self.config_dict['problem']]
        for generator, kernel_model in kernel_ensemble.items():
            tuned_metric = 0
            for tuner_name, tuner_type in FEDOT_TUNER_STRATEGY.items():
                tuning_params['tuner'] = tuner_type
                model_to_tune = deepcopy(kernel_model)
                pipeline_tuner, tuned_kernel_model = build_tuner(self,
                                                                 model_to_tune,
                                                                 tuning_params,
                                                                 kernel_data[generator],
                                                                 'head')
                if abs(pipeline_tuner.obtained_metric) > tuned_metric:
                    tuned_metric = abs(pipeline_tuner.obtained_metric)
                    self.solver = tuned_kernel_model
            tuned_kernels.update({generator: self.solver})

        return tuned_kernels

    def _kernel_strategy(self, input_data):
        kernel_ensemble, kernel_data = KernelEnsembler(self.industrial_strategy_params).transform(input_data).predict
        self.solver = self._finetune_loop(kernel_ensemble, kernel_data)

    def _federated_predict(self,
                           input_data,
                           mode: str = 'labels'):
        self.predicted_branch_probs = [x.predict(input_data).predict
                                       for x in self.solver.root_node.nodes_from]
        self.predicted_branch_labels = [np.argmax(x, axis=1) for x in self.predicted_branch_probs]
        n_samples, n_channels, n_classes = self.predicted_branch_probs[0].shape[0], \
                                           len(self.predicted_branch_probs), \
                                           self.predicted_branch_probs[0].shape[1]
        head_model = deepcopy(self.solver.root_node)
        head_model.nodes_from = []
        input_data.features = np.hstack(self.predicted_branch_labels).reshape(n_samples,
                                                                              n_channels, 1)
        head_predict = head_model.predict(self.predict_data).predict
        if mode == 'labels':
            return head_predict
        else:
            return np.argmax(head_predict, axis=1)

    def _kernel_predict(self,
                        input_data,
                        mode: str = 'labels'):
        labels_dict = {k: v.predict(input_data, mode) for k, v in self.solver.items()}
        return labels_dict

from copy import deepcopy

import numpy as np
from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from golem.core.tuning.optuna_tuner import OptunaTuner

from fedot_ind.core.ensemble.kernel_ensemble import KernelEnsembler
from fedot_ind.core.ensemble.random_automl_forest import RAFensembler
from fedot_ind.core.repository.constanst_repository import BATCH_SIZE_FOR_FEDOT_WORKER, FEDOT_WORKER_NUM, \
    FEDOT_WORKER_TIMEOUT_PARTITION, FEDOT_TUNING_METRICS, FEDOT_TUNER_STRATEGY, FEDOT_TS_FORECASTING_ASSUMPTIONS, \
    FEDOT_TASK
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
                                        'kernel_automl': self._kernel_strategy,
                                        'forecasting_assumptions': self._forecasting_strategy,
                                        'forecasting_exogenous': self._forecasting_exogenous_strategy
                                        }
        self.industrial_strategy_predict = {'federated_automl': self._federated_predict,
                                            'kernel_automl': self._kernel_predict,
                                            'forecasting_assumptions': self._forecasting_predict,
                                            'forecasting_exogenous': self._forecasting_predict}

        self.ensemble_strategy_dict = {'MeanEnsemble': np.mean,
                                       'MedianEnsemble': np.median,
                                       'MinEnsemble': np.min,
                                       'MaxEnsemble': np.max,
                                       'ProductEnsemble': np.prod}

        self.ensemble_strategy = list(self.ensemble_strategy_dict.keys())
        self.random_label = None
        self.config_dict = api_config
        self.logger = logger
        self.repo = IndustrialModels().setup_repository()
        self.kernel_ensembler = KernelEnsembler
        self.RAF_workers = None
        self.solver = None

    def fit(self, input_data):
        self.industrial_strategy_fit[self.industrial_strategy](input_data)
        return self.solver

    def predict(self, input_data, predict_mode):
        return self.industrial_strategy_predict[self.industrial_strategy](input_data, predict_mode)

    def _federated_strategy(self, input_data):
        if input_data.features.shape[0] > BATCH_SIZE_FOR_FEDOT_WORKER:
            self.logger.info('RAF algorithm was applied')

            if self.RAF_workers is None:
                batch_size = FEDOT_WORKER_NUM
            else:
                batch_size = round(input_data.features.shape[0] / self.RAF_workers)
            # batch_size = round(input_data.features.shape[0] / self.RAF_workers if self.RAF_workers
            #                                                                       is not None else FEDOT_WORKER_NUM)
            batch_timeout = round(self.config_dict['timeout'] / FEDOT_WORKER_TIMEOUT_PARTITION)
            self.config_dict['timeout'] = batch_timeout
            self.logger.info(f'Batch_size - {batch_size}. Number of batches - {self.RAF_workers}')
            self.solver = RAFensembler(composing_params=self.config_dict,
                                       n_splits=self.RAF_workers,
                                       batch_size=batch_size)
            self.logger.info(f'Number of AutoMl models in ensemble - {self.solver.n_splits}')

    def _forecasting_strategy(self, input_data):
        self.logger.info('TS forecasting algorithm was applied')
        self.config_dict['timeout'] = round(self.config_dict['timeout'] / 3)
        self.solver = {}
        for model_name, init_assumption in FEDOT_TS_FORECASTING_ASSUMPTIONS.items():
            try:
                self.config_dict['initial_assumption'] = init_assumption.build()
                industrial = Fedot(**self.config_dict)
                industrial.fit(input_data)
                self.solver.update({model_name: industrial})
            except Exception:
                self.logger.info(f'Failed during fit stage - {model_name}')

    def _forecasting_exogenous_strategy(self, input_data):
        self.logger.info('TS exogenous forecasting algorithm was applied')
        self.solver = {}
        init_assumption = PipelineBuilder().add_node('lagged', 0)
        task = FEDOT_TASK[self.config_dict['problem']]
        train_lagged, predict_lagged = train_test_data_setup(InputData(idx=np.arange(len(input_data.features)),
                                                                       features=input_data.features,
                                                                       target=input_data.features,
                                                                       task=task,
                                                                       data_type=DataTypesEnum.ts), 2)
        dataset_dict = {'lagged': train_lagged}
        exog_variable = self.industrial_strategy_params['exog_variable']
        init_assumption.add_node('exog_ts', 1)

        # Exogenous time series
        train_exog, predict_exog = train_test_data_setup(InputData(idx=np.arange(len(exog_variable)),
                                                                   features=exog_variable,
                                                                   target=input_data.features,
                                                                   task=task,
                                                                   data_type=DataTypesEnum.ts), 2)
        dataset_dict.update({f'exog_ts': train_exog})

        train_dataset = MultiModalData(dataset_dict)
        init_assumption = init_assumption.join_branches('ridge')
        self.config_dict['initial_assumption'] = init_assumption.build()

        industrial = Fedot(**self.config_dict)
        industrial.fit(train_dataset)
        self.solver = {'exog_model': industrial}

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
        self.kernel_ensembler = KernelEnsembler(self.industrial_strategy_params)
        kernel_ensemble, kernel_data = self.kernel_ensembler.transform(input_data).predict
        self.solver = self._finetune_loop(kernel_ensemble, kernel_data)
        # tuning_params = {'metric': FEDOT_TUNING_METRICS[self.config_dict['problem']], 'tuner': OptunaTuner}
        # self.solver
        # self.solver = build_tuner(self, self.solver, tuning_params, input_data, 'head')
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

    def _forecasting_predict(self,
                             input_data,
                             mode: str = True):
        labels_dict = {k: v.predict(features=input_data, in_sample=mode) for k, v in self.solver.items()}
        return labels_dict

    def _kernel_predict(self,
                        input_data,
                        mode: str = 'labels'):
        labels_dict = {k: v.predict(input_data, mode).predict for k, v in self.solver.items()}
        return labels_dict

    def _check_predictions(self, predictions):
        """Check if the predictions array has the correct size.

        Args:
            predictions: array of shape (n_samples, n_classifiers). The votes obtained by each classifier
            for each sample.

        Returns:
            predictions: array of shape (n_samples, n_classifiers). The votes obtained by each classifier
            for each sample.

        Raises:
            ValueError: if the array do not contain exactly 3 dimensions: [n_samples, n_classifiers, n_classes]

        """

        list_proba = [predictions[model_preds] for model_preds in predictions]
        transformed = []
        if self.random_label is None:
            self.random_label = {
                class_by_gen: np.random.choice(self.kernel_ensembler.classes_misses_by_generator[class_by_gen])
                for class_by_gen in self.kernel_ensembler.classes_described_by_generator}
        for prob_by_gen, class_by_gen in zip(list_proba, self.kernel_ensembler.classes_described_by_generator):
            converted_probs = np.zeros((prob_by_gen.shape[0], len(self.kernel_ensembler.all_classes)))
            for true_class, map_class in self.kernel_ensembler.mapper_dict[class_by_gen].items():
                converted_probs[:, true_class] = prob_by_gen[:, map_class]
            random_label = self.random_label[class_by_gen]
            converted_probs[:, random_label] = prob_by_gen[:, -1]
            transformed.append(converted_probs)

        return np.array(transformed).transpose((1, 0, 2))

    def ensemble_predictions(self, prediction_dict, strategy):
        transformed_predictions = self._check_predictions(prediction_dict)
        average_proba_predictions = self.ensemble_strategy_dict[strategy](transformed_predictions, axis=1)

        if average_proba_predictions.shape[1] == 1:
            average_proba_predictions = np.concatenate([average_proba_predictions, 1 - average_proba_predictions],
                                                       axis=1)

        return average_proba_predictions

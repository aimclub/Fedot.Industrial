from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.metrics.metrics_implementation import RMSE, Accuracy, F1, R2
from fedot_ind.core.repository.industrial_implementations.abstract import build_tuner
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader

BENCHMARK = 'M4'


class AbstractPipeline:

    def __init__(self, task, task_params=None, task_metric: str = None):
        self.repo = IndustrialModels().setup_repository()
        self.task = task
        self.task_params = task_params
        _metric_dict = {'classification': Accuracy,
                        'regression': RMSE,
                        'ts_forecasting': RMSE,
                        'rmse': RMSE,
                        'f1': F1,
                        'R2': R2
                        }
        if task_metric is not None:
            self.base_metric = _metric_dict[task_metric]
        else:
            self.base_metric = _metric_dict[self.task]

    def create_pipeline(self, node_list):
        pipeline = PipelineBuilder()
        if isinstance(node_list, dict):
            for branch, nodes in node_list.items():
                if isinstance(branch, int):
                    for node in nodes:
                        pipeline.add_node(node, branch_idx=branch)
                else:
                    pipeline.join_branches(nodes)
        else:
            for node in node_list:
                pipeline.add_node(node)
        return pipeline.build()

    def tune_pipeline(
            self,
            model_to_tune,
            tuning_params,
            tune_data: InputData = None):
        if tune_data is None:
            tune_data = self.train_data
        pipeline_tuner, tuned_model = build_tuner(
            self, model_to_tune, tuning_params, tune_data, 'head')
        return tuned_model

    def create_input_data(self, dataset_name):

        if self.task == 'ts_forecasting':
            train_data, _ = DataLoader(
                dataset_name=dataset_name).load_forecast_data(folder=BENCHMARK)
            target = train_data.values[-self.task_params['forecast_length']:].flatten()
            train_data = (train_data, target)
            input_train = DataCheck(
                input_data=train_data,
                task=self.task,
                task_params=self.task_params).check_input_data()
            input_test = None
        elif isinstance(dataset_name, dict):
            input_train = DataCheck(
                input_data=dataset_name['train_data'],
                task=self.task,
                task_params=self.task_params).check_input_data()
            input_test = DataCheck(
                input_data=dataset_name['test_data'],
                task=self.task,
                task_params=self.task_params).check_input_data()
        else:
            train_data, test_data = DataLoader(
                dataset_name=dataset_name).load_data()
            input_train = DataCheck(
                input_data=train_data,
                task=self.task,
                task_params=self.task_params).check_input_data()
            input_test = DataCheck(
                input_data=test_data,
                task=self.task,
                task_params=self.task_params).check_input_data()
        return input_train, input_test

    def evaluate_pipeline(self, node_list, dataset):
        test_model = self.create_pipeline(node_list)
        self.train_data, self.test_data = self.create_input_data(dataset)
        test_model.fit(self.train_data)
        if self.task == 'ts_forecasting':
            predict = test_model.predict(self.train_data)
            predict_proba = predict
            target = self.train_data.features[-self.task_params['forecast_length']:].flatten()
        else:
            predict = test_model.predict(self.test_data, 'labels')
            predict_proba = test_model.predict(self.test_data, 'probs')
            target = self.test_data.target
        metric = self.base_metric(target=target,
                                  predicted_probs=predict_proba.predict,
                                  predicted_labels=predict.predict).metric()

        return dict(fitted_model=test_model,
                    predict_labels=predict.predict,
                    predict_probs=predict_proba.predict,
                    quality_metric=metric)

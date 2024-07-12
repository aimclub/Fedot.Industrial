from typing import Union

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from pymonad.either import Either

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.metrics.metrics_implementation import RMSE, Accuracy, F1, R2
from fedot_ind.core.repository.industrial_implementations.abstract import build_tuner
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader

BENCHMARK = 'M4'


class AbstractPipeline:

    def __init__(self, task, task_params={}, task_metric: str = None):
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

    @staticmethod
    def create_pipeline(node_list, build: bool = True):
        pipeline = PipelineBuilder()

        if isinstance(node_list, dict):
            key_is_model_name = isinstance(list(node_list.keys())[0], str)
            if key_is_model_name:
                for node, params in node_list.items():
                    pipeline.add_node(node, params=params)
            else:
                for branch, nodes in node_list.items():
                    if isinstance(branch, int):
                        for node in nodes:
                            pipeline.add_node(node, branch_idx=branch)
                    else:
                        pipeline.join_branches(nodes)
        else:
            for node in node_list:
                pipeline.add_node(node)

        return pipeline.build() if build else pipeline

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
        dataset_is_dict = isinstance(dataset_name, dict)
        custom_dataset_strategy = self.task_params['industrial_strategy'] if 'industrial_strategy' \
                                                                             in self.task_params.keys() else self.task
        loader = DataLoader(dataset_name=dataset_name)

        input_train, input_test = Either(value=dataset_name,
                                         monoid=[dataset_name,
                                                 dataset_is_dict]). \
            either(left_function=loader.load_data,
                   right_function=lambda dataset_dict: loader.load_custom_data(custom_dataset_strategy))

        input_train = DataCheck(
            input_data=input_train,
            task=custom_dataset_strategy,
            task_params=self.task_params,
            industrial_task_params=None).check_input_data()
        input_test = DataCheck(
            input_data=input_test,
            task=custom_dataset_strategy,
            task_params=self.task_params,
            industrial_task_params=None).check_input_data()
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


class ApiTemplate:

    def __init__(self,
                 api_config,
                 metric_list):
        self.api_config = api_config
        self.metric_names = metric_list

    def _prepare_dataset(self, dataset):
        dataset_is_dict = isinstance(dataset, dict)
        custom_dataset_strategy = self.api_config['industrial_strategy'] if 'industrial_strategy' \
                                                                            in self.api_config.keys() \
            else self.api_config['problem'] if self.api_config['problem'] == 'ts_forecasting' else None
        loader = DataLoader(dataset_name=dataset)

        train_data, test_data = Either(value=dataset,
                                       monoid=[dataset,
                                               dataset_is_dict]). \
            either(left_function=loader.load_data,
                   right_function=lambda dataset_dict: loader.load_custom_data(custom_dataset_strategy))
        return train_data, test_data

    def _get_result(self, test_data):
        labels = self.industrial_class.predict(test_data)
        self.industrial_class.predict_proba(test_data)
        metrics = self.industrial_class.get_metrics(target=test_data[1],
                                                    rounding_order=3,
                                                    metric_names=self.metric_names)
        result_dict = dict(industrial_model=self.industrial_class,
                           labels=labels,
                           metrics=metrics)
        return result_dict

    def eval(self,
             dataset: Union[str, dict] = None,
             finetune: bool = False,
             initial_assumption: Union[list, dict] = None):
        self.train_data, self.test_data = self._prepare_dataset(dataset)
        if initial_assumption is not None:
            pipeline = AbstractPipeline.create_pipeline(initial_assumption, build=False)
            self.api_config['initial_assumption'] = pipeline
        self.industrial_class = FedotIndustrial(**self.api_config)
        Either(value=self.train_data, monoid=[dict(train_data=self.train_data,
                                                   tuning_params={'tuning_timeout': self.api_config['timeout']}),
                                              not finetune]). \
            either(left_function=lambda tuning_data: self.industrial_class.finetune(**tuning_data),
                   right_function=self.industrial_class.fit)
        return self._get_result(self.test_data)

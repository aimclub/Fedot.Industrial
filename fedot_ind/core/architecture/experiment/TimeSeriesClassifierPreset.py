import logging
from typing import List, Union
from typing import Optional

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
# from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from golem.core.tuning.sequential import SequentialTuner

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.api.utils.path_lib import DEFAULT_PATH_RESULTS
from fedot_ind.api.utils.saver_collections import ResultSaver
from fedot_ind.core.metrics.evaluation import PerformanceAnalyzer
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

np.random.seed(0)


class TimeSeriesClassifierPreset:
    """Class responsible for interaction with Fedot classifier. It allows to use FEDOT optimization
    for hyperparameters tuning and pipeline building. Nodes of the pipeline are basis functions
    from the list of branch_nodes and quantile extractor.

    Attributes:
        branch_nodes: list of nodes to be used in the pipeline
        tuning_iterations: number of iterations for tuning hyperparameters of preprocessing pipeline
        model_params: parameters of the FEDOT classification model
        dataset_name: name of the dataset to be used
        saver: object of ``ResultSaver`` class

    Notes:
        ``branch_nodes`` can be one or combination of the following: ``data_driven_basis``, ``fourier_basis``,
        ``wavelet_basis``.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.logger = logging.getLogger(f'{self.__class__.__name__}')

        self.branch_nodes: list = params.get('branch_nodes', None)
        self.model_params = params.get('model_params')
        self.available_operations = params.get('available_operations', None)
        self.dataset_name = params.get('dataset')
        self.tuning_iterations = params.get('tuning_iterations', 30)
        self.tuning_timeout = params.get('tuning_timeout', 15.0)
        self.output_folder = params.get('output_folder', DEFAULT_PATH_RESULTS)

        self.saver = ResultSaver(dataset_name=self.dataset_name,
                                 generator_name='fedot_preset',
                                 output_dir=self.output_folder)

        self.classifier = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.preprocessing_pipeline = self._build_pipeline()

        self.logger.info(f'TimeSeriesClassifierPreset initialised with [{self.branch_nodes}] nodes and '
                         f'[{self.tuning_iterations}] tuning iterations and [{self.tuning_timeout}] timeout')

    def _build_pipeline(self):
        """
        Method for building pipeline with nodes from ``branch_nodes`` list and quantile extractor.

        """
        if self.branch_nodes is None:
            self.branch_nodes = ['eigen_basis', 'fourier_basis', 'wavelet_basis']
        self.extractors = ['quantile_extractor'] * len(self.branch_nodes)

        pipeline_builder = PipelineBuilder()
        for index, (basis, extractor) in enumerate(zip(self.branch_nodes, self.extractors)):
            pipeline_builder.add_node(basis, branch_idx=index)
            pipeline_builder.add_node(extractor, branch_idx=index)
        pipeline_builder.join_branches('rf')
        return pipeline_builder.build()

    def _tune_pipeline(self, train_data: InputData):
        if train_data.num_classes > 2:
            metric = 'f1'
        else:
            metric = 'roc_auc'

        pipeline_tuner = TunerBuilder(train_data.task) \
            .with_tuner(SequentialTuner) \
            .with_metric(metric) \
            .with_timeout(self.tuning_timeout) \
            .with_iterations(self.tuning_iterations) \
            .build(train_data)
        self.preprocessing_pipeline = pipeline_tuner.tune(self.preprocessing_pipeline)

    def fit(self, features,
            target: np.ndarray = None,
            **kwargs) -> Fedot:

        with IndustrialModels():
            train_input_data = init_input_data(features, target, task='classification')

            self._tune_pipeline(train_input_data)
            self.preprocessing_pipeline.fit(train_input_data)

            self.baseline_model = self.preprocessing_pipeline.nodes[0]
            self.preprocessing_pipeline.update_node(self.baseline_model, PipelineNode('cat_features'))
            self.baseline_model.nodes_from = []
            self.preprocessing_pipeline.fit(train_input_data)

            train_output_data = self.preprocessing_pipeline.root_node.predict(train_input_data)
            train_output_data.predict = np.squeeze(train_output_data.predict)
            train_preprocessed_data = InputData(idx=train_output_data.idx,
                                                features=train_output_data.predict,
                                                target=train_output_data.target,
                                                data_type=train_output_data.data_type,
                                                task=train_output_data.task)

            self.train_features = train_output_data.features

        metric = 'roc_auc' if train_input_data.num_classes == 2 else 'f1'
        self.model_params.update({'metric': metric})
        if self.available_operations is not None:
            self.model_params.update({'available_operations': self.available_operations})
        self.classifier = Fedot(**self.model_params)
        self.classifier.fit(train_preprocessed_data)
        return self.classifier

    def predict(self, features: pd.DataFrame, target: np.array):
        _, self.prediction_label = self._predict_abstract(features, target, 'labels')
        return self.prediction_label

    def predict_proba(self, features, target):
        _, self.prediction_proba = self._predict_abstract(features, target, 'probs')
        return self.prediction_proba

    def _predict_abstract(self, features, target, mode):
        test_input_data = init_input_data(features, target)
        test_output_data = self.preprocessing_pipeline.root_node.predict(test_input_data)
        if test_output_data.features.shape[0] == 1:
            test_output_data.predict = np.squeeze(test_output_data.predict).reshape(1, -1)
        else:
            test_output_data.predict = np.squeeze(test_output_data.predict)

        test_preprocessed_data = InputData(idx=test_output_data.idx,
                                           features=test_output_data.predict,
                                           target=test_output_data.target,
                                           data_type=test_output_data.data_type,
                                           task=test_output_data.task)
        self.test_features = test_preprocessed_data.features
        prediction_baseline = self.baseline_model.predict(test_preprocessed_data).predict
        if mode == 'probs':
            prediction = self.classifier.predict_proba(test_preprocessed_data)
        else:
            prediction = self.classifier.predict(test_preprocessed_data)

        return prediction_baseline, prediction

    def get_metrics(self, target: Union[np.ndarray, pd.Series], metric_names: Union[str, List[str]]) -> dict:
        """
        Method for calculating metrics on test data.

        Args:
            target: numpy array with target values
            metric_names: list of desirable metrics names

        Returns:
            dictionary with metrics values that looks like ``{metric_name: metric_value}``

        """
        analyzer = PerformanceAnalyzer()
        # self.baseline_metrics = analyzer.calculate_metrics(target=np.ravel(target),
        #                                                    predicted_labels=self.prediction_label_baseline,
        #                                                    predicted_probs=self.prediction_proba_baseline,
        #                                                    target_metrics=metric_names)
        return analyzer.calculate_metrics(target=np.ravel(target),
                                          predicted_labels=self.prediction_label,
                                          predicted_probs=self.prediction_proba,
                                          target_metrics=metric_names)

    def save_prediction(self, predicted_data: np.ndarray, kind: str):
        self.saver.save(predicted_data, kind)

    def save_metrics(self, metrics: dict):
        # self.saver.save(self.baseline_metrics, 'baseline_metrics')
        self.saver.save(metrics, 'metrics')

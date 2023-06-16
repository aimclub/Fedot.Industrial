import logging
from typing import List, Union
from typing import Optional

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot_ind.api.utils.saver_collections import ResultSaver
from fedot_ind.core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from fedot_ind.core.architecture.utils.utils import default_path_to_save_results
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

np.random.seed(0)


class TimeSeriesClassifierPreset:
    """Class responsible for interaction with Fedot classifier. It allows to use FEDOT optimization
    for hyperparameters tuning and pipeline building. Nodes of the pipeline are basis functions
    from the list of branch_nodes and statistical extractor.

    Attributes:
        branch_nodes: list of nodes to be used in the pipeline
        model_params: parameters of the FEDOT classification model
        dataset_name: name of the dataset to be used
        output_dir: path to the directory where results will be saved
        saver: object of ``ResultSaver`` class

    Notes:
        ``branch_nodes`` can be one of the following: ``'data_driven_basis'``, ``'fourier_basis'``, ``'wavelet_basis'``.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.test_data_preprocessed = None
        self.generator_name = 'fedot_preset'
        self.branch_nodes: list = params.get('branch_nodes', ['data_driven_basis',
                                                              'fourier_basis',
                                                              'wavelet_basis'])

        self.model_params = params.get('model_params')
        self.dataset_name = params.get('dataset')
        self.output_dir = params.get('output_dir', default_path_to_save_results())

        self.saver = ResultSaver(dataset_name=self.dataset_name,
                                 generator_name=self.generator_name,
                                 output_dir=self.output_dir)
        self.logger = logging.getLogger('TimeSeriesClassifier')

        self.prediction_label = None
        self.predictor = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.input_test_data = None

        self.logger.info(f'TimeSeriesClassifierPreset initialised with [{self.branch_nodes}] nodes')

    # TODO: put some datatype
    # TODO: add multidata option
    def _init_input_data(self, X: pd.DataFrame, y: np.ndarray) -> InputData:
        """Method for initialization of InputData object from pandas DataFrame and numpy array with target values.

        Args:
            X: pandas DataFrame with features
            y: numpy array with target values

        Returns:
            InputData object convinient for FEDOT framework

        """
        input_data = InputData(idx=np.arange(len(X)),
                               features=X.values,
                               target=np.ravel(y).reshape(-1, 1),
                               task=Task(TaskTypesEnum.classification),
                               data_type=DataTypesEnum.table)

        # Multidata option

        # train_data = InputData(idx=np.arange(len(train_data[0])),
        #                        features=np.array(train_data[0].values.tolist()),
        #                        target=train_data[1].reshape(-1, 1),
        #                        task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.image)

        return input_data

    def _build_pipeline(self):
        """
        Method for building pipeline with nodes from ``branch_nodes`` list and statistical extractor.

        """
        pipeline_builder = PipelineBuilder()
        branch_idx = 0
        for node in self.branch_nodes:
            pipeline_builder.add_node(node, branch_idx=branch_idx)
            pipeline_builder.add_node('quantile_extractor', branch_idx=branch_idx)
            branch_idx += 1
        pipeline_builder.join_branches('rf')
        return pipeline_builder.build()

    def _tune_pipeline(self, pipeline: Pipeline, train_data: InputData):
        """
        Method for tuning pipeline with simultaneous tuner.

        Args:
            pipeline: pipeline to be tuned
            train_data: InputData object with train data

        Returns:
            tuned pipeline

        """
        pipeline_tuner = TunerBuilder(train_data.task) \
            .with_tuner(SimultaneousTuner) \
            .with_metric(ClassificationMetricsEnum.f1) \
            .with_iterations(30) \
            .build(train_data)
        pipeline = pipeline_tuner.tune(pipeline)
        return pipeline

    def fit(self, features,
            target: np.ndarray = None,
            **kwargs) -> object:
        """
        Method for fitting pipeline on train data. It also tunes pipeline and updates it with categorical features.

        Args:
            features: pandas DataFrame with features
            target: numpy array with target values

        Returns:
            fitted FEDOT model as object of ``Pipeline`` class

        """

        with IndustrialModels():
            self.train_data = self._init_input_data(features, target)
            self.prerpocessing_pipeline = self._build_pipeline()
            self.prerpocessing_pipeline = self._tune_pipeline(self.prerpocessing_pipeline,
                                                              self.train_data)
            self.prerpocessing_pipeline.fit(self.train_data)

            rf_node = self.prerpocessing_pipeline.nodes[0]
            self.prerpocessing_pipeline.update_node(rf_node, PipelineNode('cat_features'))
            rf_node.nodes_from = []
            rf_node.unfit()
            self.prerpocessing_pipeline.fit(self.train_data)

            train_data_preprocessed = self.prerpocessing_pipeline.root_node.predict(self.train_data)
            train_data_preprocessed.predict = np.squeeze(train_data_preprocessed.predict)

            train_data_preprocessed = InputData(idx=train_data_preprocessed.idx,
                                                features=train_data_preprocessed.predict,
                                                target=train_data_preprocessed.target,
                                                data_type=train_data_preprocessed.data_type,
                                                task=train_data_preprocessed.task)

        metric = 'roc_auc' if train_data_preprocessed.num_classes == 2 else 'f1'
        self.model_params.update({'metric': metric})
        self.predictor = Fedot(**self.model_params)

        self.predictor.fit(train_data_preprocessed)

        return self.predictor

    def predict(self, features: pd.DataFrame, target: np.array) -> dict:
        """
        Method for prediction on test data.

        Args:
            features: pandas DataFrame with features
            target: numpy array with target values

        """
        if self.test_data_preprocessed is None:
            test_data = self._init_input_data(features, target)
            test_data_preprocessed = self.prerpocessing_pipeline.root_node.predict(test_data)
            test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict)
            self.test_data_preprocessed = InputData(idx=test_data_preprocessed.idx,
                                                    features=test_data_preprocessed.predict,
                                                    target=test_data_preprocessed.target,
                                                    data_type=test_data_preprocessed.data_type,
                                                    task=test_data_preprocessed.task)

        self.prediction_label = self.predictor.predict(self.test_data_preprocessed)
        return self.prediction_label

    def predict_proba(self, features, target) -> dict:
        if self.test_data_preprocessed is None:
            test_data = self._init_input_data(features, target)
            test_data_preprocessed = self.prerpocessing_pipeline.root_node.predict(test_data)
            self.test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict)

        self.prediction_proba = self.predictor.predict_proba(self.test_data_preprocessed)
        return self.prediction_proba

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
        return analyzer.calculate_metrics(target=np.ravel(target),
                                          predicted_labels=self.prediction_label,
                                          predicted_probs=self.prediction_proba,
                                          target_metrics=metric_names)

    def save_prediction(self, predicted_data: np.ndarray, kind: str):
        self.saver.save(predicted_data, kind)

    def save_metrics(self, metrics: dict):
        self.saver.save(metrics, 'metrics')

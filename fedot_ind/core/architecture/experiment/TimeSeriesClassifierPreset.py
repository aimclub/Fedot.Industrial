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
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from golem.core.tuning.sequential import SequentialTuner

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.api.utils.path_lib import default_path_to_save_results
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
        tuning_iters: number of iterations for tuning hyperparameters of preprocessing pipeline
        model_params: parameters of the FEDOT classification model
        dataset_name: name of the dataset to be used
        saver: object of ``ResultSaver`` class

    Notes: ``branch_nodes`` can be one or combination of the following: ``'data_driven_basis'``, ``'fourier_basis'``,
           ``'wavelet_basis'``.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.branch_nodes: list = params.get('branch_nodes', None)

        self.model_params = params.get('model_params')
        self.dataset_name = params.get('dataset')
        self.tuning_iters = params.get('tuning_iterations', 30)
        self.tuning_timeout = params.get('tuning_timeout', 15.0)
        self.output_folder = params.get('output_folder', default_path_to_save_results())

        self.saver = ResultSaver(dataset_name=self.dataset_name,
                                 generator_name='fedot_preset',
                                 output_dir=self.output_folder)
        self.logger = logging.getLogger('TimeSeriesClassifier_Preset')

        self.test_predict_hash = None
        self.test_data_preprocessed = None
        self.prediction_label = None
        self.predictor = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.input_test_data = None
        self.preprocessing_pipeline = self._build_pipeline()

        self.logger.info(f'TimeSeriesClassifierPreset initialised with [{self.branch_nodes}] nodes and '
                         f'[{self.tuning_iters}] tuning iterations and [{self.tuning_timeout}] timeout')

    def _init_input_data(self, X: pd.DataFrame, y: np.ndarray) -> InputData:
        return init_input_data(X, y)

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
        # pipeline_builder.join_branches('mlp', params={'hidden_layer_sizes': (256, 128, 64, 32),
        #                                               'max_iter': 300,
        #                                               'activation': 'relu',
        #                                               'solver': 'adam', })
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
        if train_data.num_classes > 2:
            metric = ClassificationMetricsEnum.f1
        else:
            metric = ClassificationMetricsEnum.ROCAUC

        pipeline_tuner = TunerBuilder(train_data.task) \
            .with_tuner(SequentialTuner) \
            .with_metric(metric) \
            .with_timeout(self.tuning_timeout) \
            .with_iterations(self.tuning_iters) \
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

            self.preprocessing_pipeline = self._tune_pipeline(self.preprocessing_pipeline,
                                                              self.train_data)
            self.preprocessing_pipeline.fit(self.train_data)

            self.baseline_model = self.preprocessing_pipeline.nodes[0]
            self.preprocessing_pipeline.update_node(self.baseline_model, PipelineNode('cat_features'))
            self.baseline_model.nodes_from = []
            # rf_node.unfit()
            self.preprocessing_pipeline.fit(self.train_data)

            train_data_preprocessed = self.preprocessing_pipeline.root_node.predict(self.train_data)
            train_data_preprocessed.predict = np.squeeze(train_data_preprocessed.predict)

            train_data_preprocessed = InputData(idx=train_data_preprocessed.idx,
                                                features=train_data_preprocessed.predict,
                                                target=train_data_preprocessed.target,
                                                data_type=train_data_preprocessed.data_type,
                                                task=train_data_preprocessed.task)

            self.processed_train_features = train_data_preprocessed.features

        metric = 'roc_auc' if train_data_preprocessed.num_classes == 2 else 'f1'
        self.model_params.update({'metric': metric})
        self.predictor = Fedot(available_operations=['scaling',
                                                     'normalization',
                                                     #'fast_ica',
                                                     'xgboost',
                                                     'rfr',
                                                     'rf',
                                                     'logit',
                                                     'mlp',
                                                     'knn',
                                                     'lgbm',
                                                     'pca'
                                                     ]
                               , **self.model_params)

        self.predictor.fit(train_data_preprocessed)

        return self.predictor

    def predict(self, features: pd.DataFrame, target: np.array) -> dict:
        """
        Method for prediction on test data.

        Args:
            features: pandas DataFrame with features
            target: numpy array with target values

        """

        test_data = self._init_input_data(features, target)
        test_data_preprocessed = self.preprocessing_pipeline.root_node.predict(test_data)

        if test_data.features.shape[0] == 1:
            test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict).reshape(1, -1)
        else:
            test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict)
        self.test_data_preprocessed = InputData(idx=test_data_preprocessed.idx,
                                                features=test_data_preprocessed.predict,
                                                target=test_data_preprocessed.target,
                                                data_type=test_data_preprocessed.data_type,
                                                task=test_data_preprocessed.task)

        self.prediction_label_baseline = self.baseline_model.predict(self.test_data_preprocessed).predict
        self.prediction_label = self.predictor.predict(self.test_data_preprocessed)

        return self.prediction_label

    def predict_proba(self, features, target) -> dict:
        test_data = self._init_input_data(features, target)
        test_data_preprocessed = self.preprocessing_pipeline.root_node.predict(test_data)
        self.test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict)

        self.prediction_proba_baseline = self.baseline_model.predict(self.test_data_preprocessed, 'probs').predict
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
        self.baseline_metrics = analyzer.calculate_metrics(target=np.ravel(target),
                                                           predicted_labels=self.prediction_label_baseline,
                                                           predicted_probs=self.prediction_proba_baseline,
                                                           target_metrics=metric_names)
        return analyzer.calculate_metrics(target=np.ravel(target),
                                          predicted_labels=self.prediction_label,
                                          predicted_probs=self.prediction_proba,
                                          target_metrics=metric_names)

    def save_prediction(self, predicted_data: np.ndarray, kind: str):
        self.saver.save(predicted_data, kind)

    def save_metrics(self, metrics: dict):
        self.saver.save(self.baseline_metrics, 'baseline_metrics')
        self.saver.save(metrics, 'metrics')

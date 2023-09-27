import logging
from pathlib import Path
from typing import List, Union, Dict
from typing import Optional

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner
from matplotlib import pyplot as plt

from fedot_ind.api.utils.path_lib import default_path_to_save_results
from fedot_ind.api.utils.saver_collections import ResultSaver
from fedot_ind.core.metrics.evaluation import PerformanceAnalyzer
from fedot_ind.core.operation.transformation.splitter import TSTransformer
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

np.random.seed(0)


class TimeSeriesAnomalyDetectionPreset:
    """Class responsible for interaction with Fedot classifier. It allows to use FEDOT optimization
    for hyperparameters tuning and pipeline building. Nodes of the pipeline are basis functions
    from the list of branch_nodes and statistical extractor.

    Attributes:
        branch_nodes: list of nodes to be used in the pipeline
        tuning_iters: number of iterations for tuning hyperparameters of preprocessing pipeline
        model_params: parameters of the FEDOT classification model
        dataset_name: name of the dataset to be used
        output_dir: path to the directory where results will be saved
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
        self.logger = logging.getLogger('TimeSeriesAnomalyDetection_Preset')

        self.prediction_label = None
        self.auto_model = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.input_test_data = None
        self.generator = self._build_pipeline()
        self.predictor = None

        self.logger.info(f'TimeSeriesClassifierPreset initialised with [{self.branch_nodes}] nodes and '
                         f'[{self.tuning_iters}] tuning iterations and [{self.tuning_timeout}] timeout')

    def __check_multivariate_data(self, series: [np.array, List]) -> bool:
        """Method for checking if the data is multivariate.

        Args:
            series: time series

        Returns:
            True if data is multivariate, False otherwise

        """
        return isinstance(series, list) or (len(series.shape) > 1 and series.shape[1] > 1)

    def _init_input_data(self, series: np.array, anomaly_dict: Optional[Dict] = None, is_fit_stage=True) -> InputData:
        """Method for initialization of InputData object from pandas DataFrame and numpy array with target values.

        Args:
            series: numpy array with time series
            anomaly_dict: Dict with anomaly intervals
            is_fit_stage: is it fit stage or not

        Returns:
            InputData object convenient for FEDOT framework

        """
        if is_fit_stage:
            self.splitter = TSTransformer(
                strategy='frequent')
            features, target = self.splitter.transform_for_fit(series=series,
                                                               anomaly_dict=anomaly_dict,
                                                               plot=False,
                                                               binarize=False)
        else:
            features = self.splitter.transform(series=series)
            target = None

        is_multivariate_data = self.__check_multivariate_data(features)
        if is_multivariate_data:
            input_data = InputData(idx=np.arange(len(features)),
                                   features=np.array(features.tolist()),
                                   target=target.reshape(-1, 1) if target is not None else None,
                                   task=Task(TaskTypesEnum.classification),
                                   data_type=DataTypesEnum.image)
        else:
            input_data = InputData(idx=np.arange(len(features)),
                                   features=features.values,
                                   target=np.ravel(target).reshape(-1, 1) if target is not None else None,
                                   task=Task(TaskTypesEnum.classification),
                                   data_type=DataTypesEnum.table)

        return input_data

    def _build_pipeline(self):
        """
        Method for building pipeline with nodes from ``branch_nodes`` list and statistical extractor.

        """
        if self.branch_nodes is None:
            self.branch_nodes = ['eigen_basis', 'fourier_basis', 'wavelet_basis']

        self.extractors = ['quantile_extractor'] * len(self.branch_nodes)

        pipeline_builder = PipelineBuilder()

        for index, (basis, extractor) in enumerate(zip(self.branch_nodes, self.extractors)):
            pipeline_builder.add_node(basis, branch_idx=index)
            pipeline_builder.add_node(extractor, branch_idx=index)
        pipeline_builder.join_branches('mlp', params={'hidden_layer_sizes': (256, 128, 64, 32),
                                                      'max_iter': 300,
                                                      'activation': 'relu',
                                                      'solver': 'adam', })

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
            .with_tuner(SimultaneousTuner) \
            .with_metric(metric) \
            .with_timeout(self.tuning_timeout) \
            .with_iterations(self.tuning_iters) \
            .build(train_data)

        pipeline = pipeline_tuner.tune(pipeline)
        return pipeline

    def fit(self, features,
            anomaly_dict: List = None) -> object:
        """
        Method for fitting pipeline on train data. It also tunes pipeline and updates it with categorical features.

        Args:
            features: pandas DataFrame with features
            anomaly_dict: numpy array with anomaly intervals

        Returns:
            fitted FEDOT model as object of ``Pipeline`` class

        """

        with IndustrialModels():
            self.train_data = self._init_input_data(features, anomaly_dict)
            self.generator = self._tune_pipeline(self.generator,
                                                 self.train_data)
            self.generator.fit(self.train_data)
            train_data_preprocessed = self.generator.root_node.predict(self.train_data)
            train_data_preprocessed.predict = np.squeeze(train_data_preprocessed.predict)

            train_data_preprocessed = InputData(idx=train_data_preprocessed.idx,
                                                features=train_data_preprocessed.predict,
                                                target=train_data_preprocessed.target,
                                                data_type=train_data_preprocessed.data_type,
                                                task=train_data_preprocessed.task)

        metric = 'roc_auc' if train_data_preprocessed.num_classes == 2 else 'f1'
        self.model_params.update({'metric': metric})
        self.auto_model = Fedot(available_operations=['scaling',
                                                      'normalization',
                                                      'fast_ica',
                                                      'xgboost',
                                                      'rfr',
                                                      'rf',
                                                      'logit',
                                                      'mlp',
                                                      'knn',
                                                      'lgbm',
                                                      'pca'],
                                **self.model_params)

        self.auto_model.fit(train_data_preprocessed)
        self.predictor = self.auto_model.current_pipeline
        return self.auto_model

    def predict(self, features: pd.DataFrame) -> np.array:
        """
        Method for prediction on test data.

        Args:
            features: pandas DataFrame with features

        """

        test_data = self._init_input_data(features, is_fit_stage=False)
        test_data_preprocessed = self.generator.root_node.predict(test_data)

        if test_data.features.shape[0] == 1:
            test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict).reshape(1, -1)
        else:
            test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict)
        test_data_preprocessed = InputData(idx=test_data_preprocessed.idx,
                                           features=test_data_preprocessed.predict,
                                           target=test_data_preprocessed.target,
                                           data_type=test_data_preprocessed.data_type,
                                           task=test_data_preprocessed.task)

        self.prediction_label = self.auto_model.predict(test_data_preprocessed)
        self.prediction_label = self.convert_to_point_prediction(self.prediction_label, features)
        return self.prediction_label

    def predict_proba(self, features) -> np.array:
        test_data = self._init_input_data(features, is_fit_stage=False)
        test_data_preprocessed = self.generator.predict(test_data)
        test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict)
        test_data_preprocessed = InputData(idx=test_data_preprocessed.idx,
                                           features=test_data_preprocessed.predict,
                                           target=test_data_preprocessed.target,
                                           data_type=test_data_preprocessed.data_type,
                                           task=test_data_preprocessed.task)
        self.prediction_proba = self.auto_model.predict_proba(test_data_preprocessed)
        self.prediction_proba = self.convert_to_point_prediction(self.prediction_proba, features, output_type='probs')
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

    def convert_to_point_prediction(self, prediction: np.array, features: np.array,
                                    output_type: str = 'labels') -> np.array:
        """
        Converts window-like prediction to point-like prediction

        :param prediction: window-like prediction
        :param features: features array. Must have equal length to target.
        :param output_type: output_type of prediction. Possible values are "labels" or "probs"
        """
        predicted = []
        for point in range(len(features)):
            if output_type == 'labels':
                pred_for_point = prediction[point // self.splitter.freq_length].item()
            else:
                pred_for_point = prediction[point // self.splitter.freq_length]
            predicted.append(pred_for_point)
        return predicted

    def save(self, path: Path):
        """
        Saves generator and predictor pipelines in 'path' folder. Generator saves into path 'path'/generator,
        predictor into path 'path'/predictor

        :param path: path to save pipelines
        """
        self.generator.save(Path(path, 'generator'))
        self.predictor.save(Path(path, 'predictor'))

    def load(self, path: Path):
        """
        Loads generator and predictor pipelines from 'path' folder. Generator loads from path 'path'/generator,
        predictor loads from 'path'/predictor

        :param path: path to load pipelines
        """
        with IndustrialModels():
            self.generator = Pipeline().load(Path(path, 'generator', '0_pipeline_saved', '0_pipeline_saved.json'))
        self.predictor = Pipeline().load(Path(path, 'predictor', '0_pipeline_saved', '0_pipeline_saved.json'))

    def plot_prediction(self, series, true_anomalies):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 7))
        for i in ax:
            i.plot(series, color='black')
            i.set_xlabel('Time')
            i.set_ylabel('Value')
        ax[0].set_title('Real anomalies')
        ax[1].set_title('Predicted anomalies')

        unique_anomalies = np.unique(true_anomalies)
        unique_anomalies = np.setdiff1d(unique_anomalies, np.array(['no_anomaly']))

        def generate_colors(num_colors):
            colormap = plt.cm.get_cmap('tab10')
            colors = []
            for i in range(num_colors):
                colors.append(colormap(i))
            return colors

        cmap = generate_colors(len(unique_anomalies))
        color_dict = {cls: color for cls, color in zip(unique_anomalies, cmap)}

        legend_patches = [mpatches.Patch(color=color_dict[cls],
                                         label=cls) for cls in unique_anomalies]

        for anomaly_class in unique_anomalies:
            ax[0].fill_between(np.arange(len(series)), np.min(series),
                               np.max(series),
                               where=true_anomalies == anomaly_class,
                               facecolor=color_dict[anomaly_class], alpha=.4)
            ax[1].fill_between(np.arange(len(series)), np.min(series),
                               np.max(series),
                               where=np.array(self.prediction_label) == anomaly_class,
                               facecolor=color_dict[anomaly_class], alpha=.4)

        plt.legend(handles=set(legend_patches))
        plt.show()

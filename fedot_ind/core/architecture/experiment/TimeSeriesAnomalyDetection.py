from pathlib import Path
from typing import Dict, List, Union
from typing import Optional

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from matplotlib import pyplot as plt

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.core.architecture.experiment.TimeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.metrics.evaluation import PerformanceAnalyzer
from fedot_ind.core.operation.transformation.splitter import TSTransformer
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

np.random.seed(0)


class TimeSeriesAnomalyDetectionPreset(TimeSeriesClassifierPreset):
    """Class responsible for interaction with Fedot classifier. It allows to use FEDOT optimization
    for hyperparameters tuning and pipeline building. Nodes of the pipeline are basis functions
    from the list of branch_nodes and statistical extractor.

    Attributes:
        branch_nodes: list of nodes to be used in the pipeline
        tuning_iterations: number of iterations for tuning hyperparameters of preprocessing pipeline
        model_params: parameters of the FEDOT classification model
        dataset_name: name of the dataset to be used
        output_folder: path to the directory where results will be saved
        saver: object of ``ResultSaver`` class

    Notes:
        ``branch_nodes`` can be one or combination of the following: ``data_driven_basis``,
        ``fourier_basis``, ``wavelet_basis``.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.logger.info(f'{self.__class__.__name__} initialised with [{self.branch_nodes}] nodes and '
                         f'[{self.tuning_iterations}] tuning iterations and [{self.tuning_timeout}] timeout')

    def _init_input_data(self, series: np.array, anomaly_dict: Optional[Dict] = None, is_fit_stage=True) -> InputData:
        """Method for initialization of InputData object from time series and dictionary with anomaly labels.

        Args:
            series: numpy array with time series
            anomaly_dict: Dict with anomaly intervals
            is_fit_stage: is it fit stage or not

        Returns:
            InputData object convenient for FEDOT framework

        """
        if is_fit_stage:
            self.splitter = TSTransformer(strategy='frequent')
            features, self.target = self.splitter.transform_for_fit(series=series,
                                                               anomaly_dict=anomaly_dict,
                                                               plot=False,
                                                               binarize=False)
        else:
            features = self.splitter.transform(series=series)
            target = None
        features = pd.DataFrame(features)
        return init_input_data(features, self.target, task='classification')

    def fit(self, features,
            anomaly_dict: List = None) -> object:

        with IndustrialModels():
            train_input_data = self._init_input_data(features, anomaly_dict, is_fit_stage=True)

            self._tune_pipeline(train_input_data)
            self.preprocessing_pipeline.fit(train_input_data)
            self.preprocessing_pipeline.update_node(self.preprocessing_pipeline.nodes[0], PipelineNode('cat_features'))
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

    def predict(self, features) -> np.array:
        self.prediction_label = self._predict_abstract(features, 'labels')
        return self.prediction_label

    def predict_proba(self, features) -> dict:
        self.prediction_proba = self._predict_abstract(features, 'probs')
        return self.prediction_proba

    def _predict_abstract(self, features, mode):
        test_input_data = self._init_input_data(features, is_fit_stage=False)
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
        if mode == 'probs':
            prediction = self.classifier.predict_proba(test_preprocessed_data)
        else:
            prediction = self.classifier.predict(test_preprocessed_data)

        prediction = self.convert_to_point_prediction(prediction, features, output_type=mode)
        return prediction

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

        Args:
            prediction: window-like prediction
            features: features array. Must have equal length to target.
            output_type: output_type of prediction. Possible values are "labels" or "probs"

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

        Args:
            path: path to save pipelines

        """
        self.preprocessing_pipeline.save(Path(path, 'generator'))
        self.classifier.current_pipeline.save(Path(path, 'classifier'))

    def load(self, path: Path):
        """
        Loads generator and predictor pipelines from 'path' folder. Generator loads from path 'path'/generator,
        predictor loads from 'path'/predictor

        Args:
            path: path to load pipelines

        """
        with IndustrialModels():
            self.preprocessing_pipeline = Pipeline().load(Path(path, 'generator', '0_pipeline_saved', '0_pipeline_saved.json'))
        self.classifier.current_pipeline = Pipeline().load(Path(path, 'classifier', '0_pipeline_saved', '0_pipeline_saved.json'))

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
            return list(map(colormap, range(num_colors)))

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

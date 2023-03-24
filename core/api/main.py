from typing import List, Union

import numpy as np
import pandas as pd
from fedot.api.api_utils.data_definition import FeaturesType
from fedot.api.main import Fedot
from core.log import default_log as logger

from core.api.utils.method_collections import TaskGenerator
from core.api.utils.reader_collections import DataReader, YamlReader
from core.api.utils.reporter import ReporterTSC
from core.api.utils.saver_collections import ResultSaver
from core.architecture.utils.utils import default_path_to_save_results


class FedotIndustrial(Fedot):
    """
    This class is used to run Fedot in industrial mode as FedotIndustrial.

    Args:

    """

    def __init__(self,
                 input_config: Union[dict, str] = None,
                 output_folder: str = None):
        super(Fedot, self).__init__()

        self.logger = logger(self.__class__.__name__)
        self.reporter = ReporterTSC()
        self.task_pipeline_dict = {method.name: method.value for method in TaskGenerator}
        self.YAML = YamlReader()
        self.reader = DataReader()
        self.saver = ResultSaver()

        self.fitted_model = None
        self.input_config = input_config
        self.config_dict = None
        self.output_folder = output_folder

        self.__init_experiment_setup()
        self.pipeline = self.__init_pipeline()

    def __init_experiment_setup(self):
        self.logger.info('Initialising experiment setup')

        if not self.output_folder:
            self.output_folder = default_path_to_save_results()
        self.reporter.path_to_save = self.output_folder

        self.config_dict = self.YAML.init_experiment_setup(self.input_config)

    def __init_pipeline(self):
        pipeline_params = dict(generator_name=self.config_dict['feature_generator'],
                               generator_runner=self.config_dict['generator_class'],
                               model_hyperparams=self.config_dict['fedot_params'],
                               ecm_model_flag=self.config_dict['error_correction'],
                               dataset_name=self.config_dict['dataset'])

        return self.task_pipeline_dict[self.config_dict['task']](**pipeline_params)

    def fit(self,
            train_features: pd.DataFrame,
            train_target: np.ndarray,
            baseline_type: str = 'rf') -> np.ndarray:

        fitted_pipeline = self.pipeline.fit(train_features, train_target)

        return fitted_pipeline


    def predict(self,
                test_features: pd.DataFrame,
            ) -> np.ndarray:
        return self.pipeline.predict(test_features=test_features)

    def predict_proba(self,
                      test_features,
            ) -> np.ndarray:
        return self.pipeline.predict_proba(test_features=test_features)

    def get_metrics(self,
                    target: Union[np.ndarray, pd.Series] = None,
                    metric_names: Union[str, List[str]] = ('f1', 'roc_auc', 'accuracy', 'logloss', 'precision'),
                    **kwargs) -> dict:
        return self.pipeline.get_metrics(target, metric_names)

    def save_predict(self, predicted_data):
        self.pipeline.save_prediction(predicted_data)

    def load(self, path):
        """Loads saved model and Fedot graph from disk

        Args:
            path (str): path to the model
        """
        # self.pipeline.load(path)
        pass

    def plot_prediction(self, **kwargs):
        pass

    def explain(self, features: FeaturesType = None, method: str = 'surrogate_dt', visualization: bool = True,
                **kwargs) -> 'Explainer':
        return super().explain(features, method, visualization, **kwargs)


if __name__ == "__main__":
    datasets = [ 'EthanolLevel']

    for dataset_name in datasets:
        config = dict(task='ts_classification',
                      dataset=dataset_name,
                      feature_generator='spectral',
                      use_cache=False,
                      error_correction=False,
                      launches=1,
                      timeout=1,
                      n_jobs=2,
                      # wavelet_types = ['bior2.4'],
                      window_sizes = 'auto',
                      # window_sizes = [10,30],
                      )

        indus = FedotIndustrial(input_config=config, output_folder=None)
        train_data, test_data, _ = indus.reader.read(dataset_name=dataset_name)
        model = indus.fit(train_features=train_data[0], train_target=train_data[1])

        labels = indus.predict(test_features=test_data[0])
        probs = indus.predict_proba(test_features=test_data[0])
        metrics = indus.get_metrics(target=test_data[1],
                                    metric_names=['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'])

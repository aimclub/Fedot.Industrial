from typing import List, Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from core.log import default_log as logger

from core.api.utils.checkers_collections import ParameterCheck
from core.api.utils.method_collections import EnsembleGenerator, FeatureGenerator, TaskGenerator, WindowFeatureGenerator
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
                 init_config: Union[dict, str] = None,
                 output_folder: str = None):
        super(Fedot, self).__init__()

        self.logger = logger(self.__class__.__name__)
        self.reporter = ReporterTSC()
        self.feature_generator_dict = {method.name: method.value for method in FeatureGenerator}
        self.feature_generator_dict.update({method.name: method.value for method in WindowFeatureGenerator})
        self.ensemble_methods_dict = {method.name: method.value for method in EnsembleGenerator}
        self.tasks = {method.name: method.value for method in TaskGenerator}
        self.YAML = YamlReader(feature_generator=self.feature_generator_dict)
        self.reader = DataReader()
        self.checker = ParameterCheck()
        self.saver = ResultSaver()
        self.experimenter = None

        self.fitted_model = None
        self.init_config = init_config
        self.config_dict = None
        self.output_folder = output_folder

        self.__init_experiment_setup()
        self.__init_experimenter()

    def __init_experiment_setup(self):
        self.logger.info('Initialising experiment setup')

        if not self.output_folder:
            self.output_folder = default_path_to_save_results()
        self.reporter.path_to_save = self.output_folder

        self.config_dict = self.YAML.init_experiment_setup(self.init_config)

    def __init_experimenter(self):
        experiment_type = self.init_config['task']
        self.experimenter = self.tasks[experiment_type](generator_name=None,
                                                        generator_runner=None,
                                                        model_hyperparams=None,
                                                        ecm_model_flag=False,
                                                        dataset_name=None,
                                                        output_dit=self.output_folder)

    def fit(self,
            train_features: pd.DataFrame,
            target: np.ndarray,
            **kwargs) -> np.ndarray:
        return self.experimenter.fit(train_features, target)

    def predict(self,
                test_features: pd.DataFrame,
                target: np.ndarray,
                **kwargs) -> np.ndarray:
        return self.experimenter.predict(test_features, target)

    def predict_proba(self,
                      test_features,
                      target,
                      **kwargs) -> np.ndarray:
        return self.experimenter.predict_proba(test_features, target)

    def get_metrics(self,
                    target: Union[np.ndarray, pd.Series] = None,
                    metric_names: Union[str, List[str]] = None,
                    **kwargs) -> dict:
        return self.experimenter.get_metrics(target, metric_names)





if __name__ == "__main__":
    datasets = ['Car', 'UMD']

    for dataset_name in datasets:
        config = dict(task='ts_classification',
                      dataset_list=[dataset_name],
                      feature_generator=['quantile', 'wavelet'],
                      use_cache=False,
                      error_correction=False,
                      launches=1,
                      timeout=1,
                      n_jobs=2,
                      ensemble_algorithm='Rank_Ensemble')

        indus = FedotIndustrial(init_config=config, output_folder=None)
        train_data, test_data, n_classes = indus.reader.read(dataset_name=dataset_name)

        indus.fit(train_features=train_data[0], target=train_data[1])

        labels = indus.predict(test_features=test_data[0], target=test_data[1])
        probs = indus.predict_proba(test_features=test_data[0], target=test_data[1])
        metrics = indus.get_metrics(test_features=test_data[0], target=test_data[1])

        indus.pipeline.show()
        indus.explain()

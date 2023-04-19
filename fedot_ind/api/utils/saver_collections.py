import logging
import os

import pandas as pd

from fedot_ind.core.architecture.utils.utils import default_path_to_save_results


class ResultSaver:
    def __init__(self, dataset_name: str, generator_name: str, output_dir: str = None):
        if generator_name is None:
            generator_name = 'without_generator'
        self.path = self.__init_save_path(dataset_name, generator_name, output_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.save_method_dict = {'labels': self.save_labels,
                                 'probs': self.save_probs,
                                 'metrics': self.save_metrics}

    def __init_save_path(self, dataset_name, generator_name, output_dir):
        if output_dir is None:
            self.output_dir = default_path_to_save_results()
        else:
            self.output_dir = os.path.abspath(output_dir)
        path = os.path.join(self.output_dir, generator_name, dataset_name)
        os.makedirs(path, exist_ok=True)

        return path

    def save(self, predicted_data, prediction_type: str):
        self.logger.info(f'Saving predicted {prediction_type} to {self.path}')
        try:
            self.save_method_dict[prediction_type](predicted_data)
        except Exception:
            self.logger.error(f'Can not save {prediction_type} type to {self.path}')

    def save_labels(self, label_data):
        df = pd.DataFrame(label_data, dtype=int)
        df.to_csv(os.path.join(self.path, 'predicted_labels.csv'))

    def save_probs(self, prob_data):
        df_preds = pd.DataFrame(prob_data.round(3), dtype=float)
        df_preds.columns = [f'Class_{x + 1}' for x in df_preds.columns]
        df_preds.to_csv(os.path.join(self.path, 'predicted_probs.csv'))

    def save_metrics(self, metrics: dict):
        df = pd.DataFrame(metrics, index=[0])
        df.to_csv(os.path.join(self.path, 'metrics.csv'))

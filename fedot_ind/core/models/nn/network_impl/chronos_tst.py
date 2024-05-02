from typing import Optional

import numpy as np
import torch
#from chronos import ChronosPipeline
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.architecture.settings.computational import default_device
from fedot_ind.core.models.base_extractor import BaseExtractor


class ChronosExtractor:
    pass


# def chronos_small(input_dim: int = 1,
#                   seq_len: int = 1,
#                   num_features: int = 100):
#     model = ChronosPipeline.from_pretrained("amazon/chronos-t5-small",
#                                             device_map='cpu',
#                                             torch_dtype=torch.bfloat16)
#     chronos_encoder = model.model.model.encoder
#     return chronos_encoder


# class ChronosExtractor(BaseExtractor):
#     """Feature space generator based on Chronos model.
#
#     Attributes:
#         num_features: int, the number of features.
#
#     Example:
#         To use this operation you can create pipeline as follows::
#             from fedot.core.pipelines.pipeline_builder import PipelineBuilder
#             from examples.fedot.fedot_ex import init_input_data
#             from fedot_ind.tools.loader import DataLoader
#             from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
#
#             train_data, test_data = DataLoader(dataset_name='Ham').load_data()
#             with IndustrialModels():
#                 pipeline = PipelineBuilder().add_node('chronos_extractor')\
#                                             .add_node('rf').build()
#                 input_data = init_input_data(train_data[0], train_data[1])
#                 pipeline.fit(input_data)
#                 features = pipeline.predict(input_data)
#                 print(features)
#
#     """
#
#     def __init__(self, params: Optional[OperationParameters] = None):
#         super().__init__(params)
#         self.num_features = params.get('num_features', 10000)
#
#     def __repr__(self):
#         return 'TransformerFeatureSpace'
#
#     def _save_and_clear_cache(self, model_list: list):
#         del model_list
#         with torch.no_grad():
#             torch.cuda.empty_cache()
#
#     def _generate_features_from_ts(self, ts: np.array, mode: str = 'multivariate'):
#
#         if ts.shape[1] > 1 and mode == 'chanel_independent':
#             chrono_model = chronos_small(input_dim=1,
#                                          seq_len=ts.shape[2],
#                                          num_features=self.num_features)
#
#             n_dim = range(ts.shape[1])
#             ts_converted = [ts[:, i, :] for i in n_dim]
#             ts_converted = [x.reshape(x.shape[0], 1, x.shape[1])
#                             for x in ts_converted]
#             model_list = [chrono_model for i in n_dim]
#         else:
#             chrono_model = chronos_small(input_dim=ts.shape[1],
#                                          seq_len=ts.shape[2],
#                                          num_features=self.num_features)
#
#             ts_converted = [ts.swapaxes(1, 2)]
#             model_list = [chrono_model]
#
#         features = [chrono_model(inputs_embeds=torch.Tensor(data).to(default_device('cpu')).to(torch.long))
#                     for model, data in zip(model_list, ts_converted)]
#
#         chrono_features = [feature_by_dim.swapaxes(1, 2) for feature_by_dim in features]
#         minirocket_features = np.concatenate(chrono_features, axis=1)
#         minirocket_features = OutputData(idx=np.arange(minirocket_features.shape[2]),
#                                          task=self.task,
#                                          predict=minirocket_features,
#                                          data_type=DataTypesEnum.image)
#         self._save_and_clear_cache(model_list)
#         return minirocket_features
#
#     def generate_chronos_features(self, ts: np.array) -> InputData:
#         return self._generate_features_from_ts(ts)
#
#     def generate_features_from_ts(self, ts_data: np.array,
#                                   dataset_name: str = None):
#         return self.generate_chronos_features(ts=ts_data)
#
#     def _transform(self,
#                    input_data: InputData) -> np.array:
#         """
#         Method for feature generation for all series
#         """
#         self.task = input_data.task
#         self.task.task_params = self.__repr__()
#         feature_matrix = self.generate_features_from_ts(input_data.features)
#         feature_matrix.predict = self._clean_predict(feature_matrix.predict)
#         return feature_matrix

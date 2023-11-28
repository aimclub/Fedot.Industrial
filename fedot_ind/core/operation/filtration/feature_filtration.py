from typing import Optional

import numpy as np
import pandas as pd
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from scipy.spatial.distance import cdist
from scipy.stats import stats

from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector


class FeatureFilter(IndustrialCachableOperationImplementation):
    def __int__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)

    def _init_params(self):
        self.grouping_level = 0.4
        self.fourier_approx = 'exact'
        self.explained_dispersion = 0.9
        self.method_dict = {'EigenBasisImplementation': self.filter_dimension_num,
                            'FourierBasisImplementation': self.filter_signal,
                            'LargeFeatureSpace': self.filter_feature_num}

    def _transform(self, operation):
        self._init_params()
        operation_name = operation.task.task_params
        if operation_name in self.method_dict.keys():
            method = self.method_dict[operation_name]
            return method(operation)
        else:
            return operation.features

    def filter_dimension_num(self, data):
        if len(data.features.shape) < 3:
            grouped_components = [self._compute_component_corr(data.features)]
        else:
            grouped_components = list(map(self._compute_component_corr, data.features))
        dimension_distrib = [x.shape[0] for x in grouped_components]
        dominant_dim = stats.mode(dimension_distrib).mode
        grouped_predict = [x[:dominant_dim, :] for x in grouped_components]
        return np.stack(grouped_predict) if len(grouped_predict) > 1 else grouped_predict[0]

    def _compute_component_corr(self, sample):
        component_idx_list = list(range(sample.shape[0]))
        del component_idx_list[0]
        if len(component_idx_list) == 1:
            return sample
        else:
            grouped_predict = sample[0, :].reshape(1, -1)
            tmp = pd.DataFrame(sample[1:, :])
            component_list = []
            correlation_matrix = cdist(metric='cosine', XA=tmp.values, XB=tmp.values)
            if (correlation_matrix > self.grouping_level).sum() > 0:
                for index in component_idx_list:
                    if len(component_idx_list) == 0:
                        break
                    else:
                        component_idx_list.remove(index)
                        for correlation_level, component in zip(correlation_matrix, sample[1:, :]):
                            if len(component_idx_list) == 0:
                                break
                            grouped_v = component
                            for cor_level in correlation_level[index:]:
                                if cor_level > self.grouping_level:
                                    component_idx = np.where(correlation_level == cor_level)[0][0] + 1
                                    grouped_v = grouped_v + sample[component_idx, :]
                                    component_idx_list.remove(component_idx)
                            component_list.append(grouped_v)
                    component_list = [x.reshape(1, -1) for x in component_list]
                    grouped_predict = np.concatenate([grouped_predict, *component_list], axis=0)
                return grouped_predict
            else:
                return sample

    def filter_feature_num(self, data):
        model = PipelineNode('pca', params={'n_components': self.explained_dispersion})
        return model.fit(data).predict

    def filter_signal(self, data):
        dominant_window_size = WindowSizeSelector(method='dff').get_window_size(data)
        model = np.median(data) + FourierBasisImplementation(
            params={'threshold': dominant_window_size,
                    'approximation': self.fourier_approx}). \
            transform(data).features
        return model

from typing import Optional

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.repository.constanst_repository import KERNEL_BASELINE_FEATURE_GENERATORS
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


class TabularExtractor(BaseExtractor):
    """Class responsible for quantile feature generator experiment.

    Attributes:
        window_size (int): size of window
        stride (int): stride for window
        var_threshold (float): threshold for variance

    Example:
        To use this class you need to import it and call needed methods::

            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('quantile_extractor',
                                                       params={'window_size': 20, 'window_mode': True})
                                            .add_node('rf')
                                            .build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.feature_domain = params.get('feature_domain', 'all')
        self.feature_params = params.get('feature_params', {})
        self.explained_dispersion = params.get('explained_dispersion', .975)
        self.reduce_dimension = params.get('reduce_dimension', True)

        self.repo = IndustrialModels().setup_repository()
        self.pca_is_fitted = False
        self.scaler = StandardScaler()
        self.pca = PCA(self.explained_dispersion)

    def _reduce_dim(self, features, target):
        if self.pca_is_fitted:
            return self.pca.transform(self.scaler.transform(features))
        else:
            self.pca_is_fitted = True
            return self.pca.fit_transform(self.scaler.fit_transform(features, target))

    def create_feature_matrix(self, feature_list: list):
        return np.concatenate([x.reshape(x.shape[0], x.shape[1] * x.shape[2])
                               for x in feature_list], axis=1).squeeze()

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """

        feature_list = self.generate_features_from_ts(input_data)
        self.predict = self.create_feature_matrix(feature_list)
        return self.predict if not self.reduce_dimension else self._reduce_dim(self.predict, input_data.target)

    def generate_features_from_ts(self,
                                  input_data: InputData,
                                  window_length: int = None) -> InputData:
        feature_domain_models = [model for model in KERNEL_BASELINE_FEATURE_GENERATORS]
        self.feature_list = []

        if not self.feature_domain.__contains__('all'):
            feature_domain_models = [model for model in feature_domain_models
                                     if model.__contains__(self.feature_domain)]

        for model_name in feature_domain_models:
            model = KERNEL_BASELINE_FEATURE_GENERATORS[model_name]
            model.heads[0].parameters['use_sliding_window'] = self.use_sliding_window
            model = model.build()
            self.feature_list.append(model.fit(input_data).predict)

        return self.feature_list

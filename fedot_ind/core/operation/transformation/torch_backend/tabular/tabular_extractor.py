from typing import Optional
import torch

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from pymonad.either import Either

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.data.park_transformation import park_transform
from fedot_ind.core.repository.constanst_repository import KERNEL_BASELINE_FEATURE_GENERATORS
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


class PCA_transformation:
    """Class for PCA transformation.

    Attributes:
        self.n_components: int, number of principle components.
        self.explained_variance: float, explained variance.
        self.mean: float, the mean value of matrix.
        self.components: tensor, matrix for PCA transformation.
        self.fitted: bool, if False, then required to be fitted.
    """

    def __init__(
            self,
            n_components=None,
            explained_variance=0.975):
        super().__init__()
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.mean = None
        self.components = None
        self.fitted = False

    def fit(self, X: torch.Tensor):
        X = X - X.mean()
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        if self.n_components is None:
            varience = (S ** 2) / (S ** 2).sum()
            cumsum = torch.cumsum(varience, dim=0)
            k = (cumsum < self.explained_variance).sum() + 1
        else:
            k = self.n_components
        self.components = V[:k]
        self.fitted = True
        return self

    def forward(self, X: torch.Tensor):
        assert self.fitted, "Not fitted"
        X = X - X.mean()
        return X @ self.components.T


class TabularExtractorTorch(BaseExtractor):
    """Class responsible for statistical feature generator experiment.

    Attributes:
        window_size (int): size of window
        stride (int): stride for window
        var_threshold (float): threshold for variance
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.feature_domain = params.get('feature_domain', 'all')
        self.feature_params = params.get('feature_params', {})
        self.explained_dispersion = params.get('explained_dispersion', .975)
        self.reduce_dimension = params.get('reduce_dimension', True)

        self.repo = IndustrialModels().setup_repository()
        self.custom_tabular_transformation = {'park_transformation': park_transform}
        self.pca_is_fitted = False
        self.pca = PCA_transformation(explained_variance=self.explained_dispersion)

    def _reduce_dim(self, features: torch.Tensor):
        if self.pca_is_fitted:
            return self.pca.forward(features)
        else:
            self.pca_is_fitted = True
            self.pca.fit(features)
            return self.pca.forward(features)

    def _create_from_custom_fg(self, input_data):
        for model_name, nodes in self.feature_domain.items():
            if model_name.__contains__('custom'):
                transform_method = self.custom_tabular_transformation[nodes[0]]
                ts_representation = transform_method(input_data)
            else:
                model = PipelineBuilder()
                for node in nodes:
                    if isinstance(node, tuple):
                        model.add_node(operation_type=node[0], params=node[1])
                    else:
                        model.add_node(operation_type=node)
                model = model.build()
                ts_representation = model.fit(input_data).predict
            self.feature_list.append(ts_representation)

    def _create_from_default_fg(self, input_data):
        feature_domain_models = [model for model in KERNEL_BASELINE_FEATURE_GENERATORS]

        if not self.feature_domain.__contains__('all'):
            feature_domain_models = [model for model in feature_domain_models
                                     if model.__contains__(self.feature_domain)]

        for model_name in feature_domain_models:
            model = KERNEL_BASELINE_FEATURE_GENERATORS[model_name]
            model.heads[0].parameters['use_sliding_window'] = self.use_sliding_window
            model = model.build()
            self.feature_list.append(model.fit(input_data).predict)

    def create_feature_matrix(self, feature_list: list) -> torch.Tensor:
        [torch.from_numpy(x).reshape(x.shape[0], x.shape[1] * x.shape[2]) for x in feature_list]
        return torch.cat([torch.from_numpy(x).reshape(x.shape[0], x.shape[1] * x.shape[2])
                          for x in feature_list], dim=1).squeeze()

    def _transform(self, input_data: InputData) -> torch.Tensor:
        """
        Method for feature generation for all series
        """

        feature_list = self.generate_features_from_ts(input_data)
        self.predict = self.create_feature_matrix(feature_list)
        return self.predict if not self.reduce_dimension else self._reduce_dim(self.predict)

    def generate_features_from_ts(self,
                                  input_data: InputData,
                                  window_length: int = None) -> InputData:
        is_custom_feature_representation = isinstance(self.feature_domain, dict)
        self.feature_list = []
        Either(value=input_data,
               monoid=[input_data,
                       is_custom_feature_representation]).either(left_function=self._create_from_default_fg,
                                                                 right_function=self._create_from_custom_fg)
        return self.feature_list

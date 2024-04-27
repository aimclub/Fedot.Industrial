from typing import Optional

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils import mean_covariance
from pyriemann.utils.distance import distance
from sklearn.utils.extmath import softmax

from fedot_ind.core.models.base_extractor import BaseExtractor


class RiemannExtractor(BaseExtractor):
    """Class responsible for riemann tangent space features generator.

    Attributes:
        estimator (str): estimator for covariance matrix, 'corr', 'cov', 'lwf', 'mcd', 'hub'
        distance_metric (str): metric for tangent space, 'riemann', 'logeuclid', 'euclid'

    Example:
        To use this class you need to import it and call needed methods::

            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('riemann_extractor')
                                            .add_node('rf')
                                            .build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.classes_ = None
        extraction_dict = {'mdm': self.extract_centroid_distance,
                           'tangent': self.extract_riemann_features,
                           'ensemble': self._ensemble_features}

        self.n_filter = params.get('nfilter', 2)
        self.estimator = params.get('estimator', 'scm')
        self.covariance_metric = params.get('SPD_metric', 'riemann')
        self.distance_metric = params.get('tangent_metric', 'riemann')
        self.extraction_strategy = params.get(
            'extraction_strategy ', 'ensemble')

        self.covariance_transformer = params.get('SPD_space', None)
        self.tangent_projector = params.get('tangent_space', None)
        if np.any([self.covariance_transformer, self.tangent_projector]) is None:
            self._init_spaces()
            self.fit_stage = True
        self.extraction_func = extraction_dict[self.extraction_strategy]

        self.logging_params.update({
            'estimator': self.estimator,
            'tangent_space_metric': self.distance_metric,
            'SPD_space_metric': self.covariance_metric})

    def _init_spaces(self):
        self.covariance_transformer = Covariances(estimator='scm')
        self.tangent_projector = TangentSpace(metric=self.distance_metric)
        self.shrinkage = Shrinkage()

    def extract_riemann_features(self, input_data: InputData) -> InputData:
        if not self.fit_stage:
            SPD = self.covariance_transformer.transform(input_data.features)
            SPD = self.shrinkage.transform(SPD)
            ref_point = self.tangent_projector.transform(SPD)
        else:
            SPD = self.covariance_transformer.fit_transform(
                input_data.features, input_data.target)
            SPD = self.shrinkage.fit_transform(SPD)
            ref_point = self.tangent_projector.fit_transform(SPD)
            self.fit_stage = False
            self.classes_ = np.unique(input_data.target)
        return ref_point

    def extract_centroid_distance(self, input_data: InputData):
        input_data.target = input_data.target.astype(int)
        if self.fit_stage:
            SPD = self.covariance_transformer.fit_transform(
                input_data.features, input_data.target)
            SPD = self.shrinkage.transform(SPD)

        else:
            SPD = self.covariance_transformer.transform(input_data.features)
            SPD = self.shrinkage.fit_transform(SPD)

        self.covmeans_ = [mean_covariance(SPD[np.array(input_data.target == ll).flatten()],
                                          metric=self.covariance_metric) for ll in self.classes_]

        n_centroids = len(self.covmeans_)
        dist = [distance(SPD, self.covmeans_[m], self.distance_metric)
                for m in range(n_centroids)]
        dist = np.concatenate(dist, axis=1)
        feature_matrix = softmax(-dist ** 2)
        return feature_matrix

    def _ensemble_features(self, input_data: InputData):
        tangent_features = self.extract_riemann_features(input_data)
        dist_features = self.extract_centroid_distance(input_data)
        feature_matrix = np.concatenate(
            [tangent_features, dist_features], axis=1)
        return feature_matrix

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """

        feature_matrix = self.extraction_func(input_data)
        self.predict = self._clean_predict(feature_matrix)
        return self.predict

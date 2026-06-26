from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils import mean_covariance
from pyriemann.utils.distance import distance
from sklearn.utils.extmath import softmax

from fedot_ind.core.architecture.abstraction.decorators import convert_to_3d_torch_array
from fedot_ind.core.models.base_extractor import BaseExtractor


class RiemannExtractor(BaseExtractor):
    """Class responsible for riemann tangent space features generator.

    Attributes:
        estimator (str): estimator for covariance matrix ('corr', 'cov', 'scm', 'lwf', 'mcd', 'hub')
        spd_metric (str): metric for SPD manifold distance ('riemann', 'logeuclid', 'euclid')
        tangent_metric (str): metric for tangent space ('riemann', 'logeuclid', 'euclid')
        extraction_strategy (str): feature extraction approach ('mdm', 'tangent', 'ensemble')

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
        params = params or {}

        self.estimator = params.get('estimator', 'scm')
        self.spd_metric = params.get('SPD_metric', 'riemann')
        self.tangent_metric = params.get('tangent_metric', 'riemann')
        self.extraction_strategy = params.get(
            'extraction_strategy',
            params.get('extraction_method', 'ensemble'),
        )

        self.spd_space = Covariances(estimator=self.estimator)
        self.shrinkage = Shrinkage()
        self.tangent_space = TangentSpace(metric=self.tangent_metric)

        self.classes_ = None
        self.covmeans_ = None
        self.is_fitted = False
        self.predict = None

        self._validate_params()
        self.logging_params.update({
            'estimator': self.estimator,
            'tangent_metric': self.tangent_metric,
            'SPD_metric': self.spd_metric,
            'extraction_strategy': self.extraction_strategy,
        })

    def __repr__(self):
        return 'Riemann Manifold Class for TS representation'

    def _validate_params(self):
        valid_strategies = {'mdm', 'tangent', 'ensemble'}
        if self.extraction_strategy not in valid_strategies:
            raise ValueError(
                f"Unsupported extraction strategy: '{self.extraction_strategy}'. "
                f"Valid options are: {valid_strategies}"
            )

        valid_estimators = {'corr', 'cov', 'scm', 'lwf', 'oas', 'mcd', 'hub'}
        if self.estimator not in valid_estimators:
            raise ValueError(
                f"Unsupported estimator: '{self.estimator}'. "
                f"Valid options are: {valid_estimators}"
            )

        valid_metrics = {'riemann', 'logeuclid', 'euclid', 'logdet', 'kullback', 'wasserstein'}
        if self.spd_metric not in valid_metrics:
            raise ValueError(
                f"Unsupported SPD_metric: '{self.spd_metric}'. "
                f"Valid options are: {valid_metrics}"
            )
            
        if self.tangent_metric not in valid_metrics:
            raise ValueError(
                f"Unsupported tangent_metric: '{self.tangent_metric}'. "
                f"Valid options are: {valid_metrics}"
            )

    def _prepare_tensor(self, x: np.ndarray) -> np.ndarray:
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        return x

    def fit(self, input_data: InputData):
        """Called ONCE at start. Trains everything."""

        X = self._prepare_tensor(input_data.features)
        y = np.asarray(input_data.target).flatten() if input_data.target is not None else None

        SPD = self.spd_space.fit_transform(X)
        SPD = self.shrinkage.fit_transform(SPD)

        if self.extraction_strategy in ['tangent', 'ensemble']:
            self.tangent_space.fit(SPD, y)

        if self.extraction_strategy in ['mdm', 'ensemble']:
            if y is None or len(y) == 0:
                raise ValueError("Target data is required to fit MDM centroids.")
            self.classes_ = np.unique(y)
            self.covmeans_ = [mean_covariance(SPD[np.array(y == ll).flatten()],
                                              metric=self.spd_metric) for ll in self.classes_]
        self.is_fitted = True
        return self
    
    @convert_to_3d_torch_array
    def _transform(self, input_data: InputData) -> OutputData:
        
        if not self.is_fitted:
            self.fit(input_data)
        
        X = self._prepare_tensor(input_data.features)

        SPD = self.spd_space.transform(X)
        SPD = self.shrinkage.transform(SPD)

        features = []

        if self.extraction_strategy in {'tangent', 'ensemble'}:
            tangent_features = self.tangent_space.transform(SPD)
            features.append(tangent_features)

        if self.extraction_strategy in {'mdm', 'ensemble'}:
            n_centroids = len(self.covmeans_)
            distances = [
                distance(SPD, self.covmeans_[m], metric=self.spd_metric) 
                for m in range(n_centroids)
            ]
            dist_matrix = np.column_stack(distances)
            
            mdm_features = softmax(-dist_matrix ** 2)
            features.append(mdm_features)

        if len(features) > 1:
            feature_matrix = np.concatenate(features, axis=1)
        else:
            feature_matrix = features[0]

        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        elif feature_matrix.ndim > 2:
            feature_matrix = feature_matrix.reshape(feature_matrix.shape[0], -1)

        if not np.isfinite(feature_matrix).all():
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        self.predict = self._clean_predict(feature_matrix)
        return self.predict

    
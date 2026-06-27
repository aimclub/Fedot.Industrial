from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils import mean_covariance, median_riemann, median_euclid
from pyriemann.utils.distance import distance
from sklearn.utils.extmath import softmax
import warnings

from fedot_ind.core.architecture.abstraction.decorators import convert_to_3d_torch_array
from fedot_ind.core.models.base_extractor import BaseExtractor

class RiemannExtractor(BaseExtractor):
    """Class responsible for riemann tangent space features generator.

    Attributes:
        estimator (str): estimator for covariance matrix ('corr', 'cov', 'scm', 'lwf', 'mcd', 'hub')
        spd_metric (str): metric for SPD manifold distance ('riemann', 'logeuclid', 'euclid')
        tangent_metric (str): metric for tangent space ('riemann', 'logeuclid', 'euclid')
        extraction_strategy (str): feature extraction approach ('mdm', 'tangent', 'ensemble')
        centroid_strategy (str): strategy for centroid calculation ('class-wise', 'global')
        centroid_type (str): type of centroid ('mean', 'median')

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

        default_centroid_strategy = 'global' if self.extraction_strategy == 'tangent' else 'class-wise'
        self.centroid_strategy = params.get('centroid_strategy', default_centroid_strategy)
        self.centroid_type = params.get('centroid_type', 'mean')

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
            'centroid_strategy': self.centroid_strategy,
            'centroid_type': self.centroid_type,
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
        
        valid_centroid_strategies = {'class-wise', 'global'}
        if self.centroid_strategy not in valid_centroid_strategies:
            raise ValueError(
                f"Unsupported centroid_strategy: '{self.centroid_strategy}'. "
                f"Valid options are: {valid_centroid_strategies}"
            )
        
        valid_centroid_types = {'mean', 'median'}
        if self.centroid_type not in valid_centroid_types:
            raise ValueError(
                f"Unsupported centroid_type: '{self.centroid_type}'. "
                f"Valid options are: {valid_centroid_types}. Mean is used for L2 metrics, median is used for L1 metrics."
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
        
        if self.centroid_type == 'median' and self.spd_metric not in {'riemann', 'euclid'}:
            raise ValueError(
                f"Robust strategy (centroid_type='median') is only natively supported "
                f"for 'riemann' and 'euclid' metrics in pyriemann. "
                f"Received SPD_metric: '{self.spd_metric}'."
            )    
    
        if self.extraction_strategy == 'tangent':
            if self.centroid_strategy == 'class-wise':
                warnings.warn(
                    "Conceptual mismatch: extraction_strategy is 'tangent', but centroid_strategy is explicitly "
                    "set to 'class-wise'. TangentSpace always projects into a single space based on a global centroid. "
                    "The 'class-wise' setting will be completely ignored.",
                    UserWarning
                )
            if self.centroid_type == 'median':
                warnings.warn(
                    "Methodology mismatch: extraction_strategy is 'tangent', but centroid_type is 'median'. "
                    "The TangentSpace class strictly computes the classic mean Frechet centroid under the hood. "
                    "Your request for a robust 'median' centroid will be ignored.",
                    UserWarning
                )

    def _prepare_tensor(self, x: np.ndarray) -> np.ndarray:
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        return x

    def _calculate_centroid(self, covmats: np.ndarray) -> np.ndarray:
            if self.centroid_type == 'mean':
                return mean_covariance(covmats, metric=self.spd_metric)
            elif self.centroid_type == 'median':
                if self.spd_metric == 'riemann':
                    return median_riemann(covmats)
                elif self.spd_metric == 'euclid':
                    return median_euclid(covmats)
                else:
                    raise ValueError(
                        f"Median calculation is not natively supported for metric '{self.spd_metric}'. "
                        "Use 'riemann' or 'euclid', or change centroid_type to 'mean'."
                    )

    def fit(self, input_data: InputData):
        """Called ONCE at start. Trains everything."""

        X = self._prepare_tensor(input_data.features)
        y = np.asarray(input_data.target).flatten() if input_data.target is not None else None

        SPD = self.spd_space.fit_transform(X)
        SPD = self.shrinkage.fit_transform(SPD)

        if self.extraction_strategy in ['tangent', 'ensemble']:
            self.tangent_space.fit(SPD, y)

        if self.extraction_strategy in ['mdm', 'ensemble']:
            if self.centroid_strategy == 'class-wise':
                if y is None or len(y) == 0:
                    raise ValueError("Target data is required to fit MDM centroids.")
                self.classes_ = np.unique(y)
                self.covmeans_ = [self._calculate_centroid(SPD[y == ll]) for ll in self.classes_]

            elif self.centroid_strategy == 'global':
                self.covmeans_ = [self._calculate_centroid(SPD)]
        self.is_fitted = True
        return self
    
    @convert_to_3d_torch_array
    def _transform(self, input_data: InputData) -> OutputData:
        
        if not self.is_fitted:  
            warnings.warn(
                "RiemannExtractor is not fitted. Calling 'fit' inside 'transform' with provided input data. "
                "Warning: If this is test data, it may cause data leakage.",
                UserWarning
            )
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
            mdm_features = np.column_stack(distances)
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

    
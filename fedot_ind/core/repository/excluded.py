from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import \
    LabelEncodingImplementation, OneHotEncodingImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_filters import \
    IsolationForestClassImplementation, IsolationForestRegImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_selectors import \
    LinearClassFSImplementation, NonLinearClassFSImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import *
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    ExogDataTransformationImplementation
from fedot.core.operations.evaluation.operation_implementations.models.knn import FedotKnnClassImplementation, \
    FedotKnnRegImplementation
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression as SklearnLinReg,
)
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB, MultinomialNB as SklearnMultinomialNB

from fedot_ind.core.models.manifold.riemann_embeding import RiemannExtractor
from fedot_ind.core.models.nn.network_impl.dummy_nn import DummyOverComplicatedNeuralNetwork
from fedot_ind.core.models.nn.network_impl.explainable_convolution_model import XCModel
from fedot_ind.core.models.nn.network_impl.lora_nn import LoraModel
from fedot_ind.core.models.nn.network_impl.omni_scale import OmniScaleModel
from fedot_ind.core.models.nn.network_impl.tst import TSTModel
from fedot_ind.core.operation.dummy.dummy_operation import DummyOperation
from fedot_ind.core.operation.filtration.feature_filtration import FeatureFilter

EXCLUDED_OPERATION_MUTATION = {
    'regression': ['recurrence_extractor',
                   'lora_model',
                   'topological_extractor',
                   "nbeats_model",
                   'tcn_model',
                   'dummy',
                   'deepar_model'
                   ],
    'anomaly_detection': ['inception_model',
                          'resnet_model',
                          'recurrence_extractor',
                          'xgbreg',
                          'sgdr',
                          'kernel_pca',
                          'resample',
                          'inception_model',
                          'simple_imputation',
                          'channel_filtration',
                          'recurrence_extractor',
                          'quantile_extractor',
                          'riemann_extractor',
                          'minirocket_extractor',
                          'treg',
                          'knnreg',
                          'resnet_model',
                          'dtreg'
                          ],
    'ts_forecasting': [
        'xgbreg',
        'sgdr',
        'kernel_pca',
        'resample',
        'inception_model',
        'simple_imputation',
        'channel_filtration',
        'recurrence_extractor',
        'quantile_extractor',
        'riemann_extractor',
        'minirocket_extractor',
        'treg',
        'knnreg',
        'resnet_model',
        'dtreg'
    ],
    'classification': [
        'resnet_model',
        'knnreg',
        'recurrence_extractor',
        'bernb',
        'qda',
    ]}

TEMPORARY_EXCLUDED = {
    'INDUSTRIAL_CLF_PREPROC_MODEL': {
        'rfe_lin_class': LinearClassFSImplementation,
        'rfe_non_lin_class': NonLinearClassFSImplementation,
    },
    'FEDOT_PREPROC_MODEL': {'pca': PCAImplementation,
                            'fast_ica': FastICAImplementation,
                            'poly_features': PolyFeaturesImplementation,
                            'exog_ts': ExogDataTransformationImplementation,
                            # categorical encoding
                            'one_hot_encoding': OneHotEncodingImplementation,
                            'label_encoding': LabelEncodingImplementation
                            },
    'FORECASTING_PREPROC': {'exog_ts': ExogDataTransformationImplementation},
    'INDUSTRIAL_PREPROC_MODEL': {
        'cat_features': DummyOperation,
        'dimension_reduction': FeatureFilter,
        # 'signal_extractor': SignalExtractor,
        # isolation_forest forest
        'isolation_forest_class': IsolationForestClassImplementation,
        'isolation_forest_reg': IsolationForestRegImplementation,
        # 'chronos_extractor': ChronosExtractor,
        'riemann_extractor': RiemannExtractor,
    },
    'SKLEARN_REG_MODELS': {
        'gbr': GradientBoostingRegressor,
        'rfr': RandomForestRegressor,
        'adareg': AdaBoostRegressor,
        'linear': SklearnLinReg,
        'knnreg': FedotKnnRegImplementation,
    },
    'SKLEARN_CLF_MODELS': {'bernb': SklearnBernoulliNB,
                           'multinb': SklearnMultinomialNB,
                           'knn': FedotKnnClassImplementation
                           },
    'NEURAL_MODELS': {
        # linear_dummy_model
        'dummy': DummyOverComplicatedNeuralNetwork,
        # linear_dummy_model
        'lora_model': LoraModel,
        'omniscale_model': OmniScaleModel,
        # transformer models
        'tst_model': TSTModel,
        # explainable models
        'xcm_model': XCModel
    }
}

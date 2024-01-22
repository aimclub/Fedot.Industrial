from enum import Enum
from itertools import chain

from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.topological_extractor import \
    TopologicalFeaturesImplementation
from fedot.core.operations.evaluation.operation_implementations.models.knn import FedotKnnClassImplementation, \
    FedotKnnRegImplementation

from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB, MultinomialNB as SklearnMultinomialNB
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, \
    RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import (
    Lasso as SklearnLassoReg,
    LinearRegression as SklearnLinReg,
    LogisticRegression as SklearnLogReg,
    Ridge as SklearnRidgeReg,
    SGDRegressor as SklearnSGD
)
from fedot_ind.core.models.nn.network_impl.mini_rocket import MiniRocketExtractor
from fedot_ind.core.models.recurrence.reccurence_extractor import RecurrenceExtractor
from fedot_ind.core.models.signal.signal_extractor import SignalExtractor
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.topological.topological_extractor import TopologicalExtractor
from fedot_ind.core.operation.dummy.dummy_operation import DummyOperation
from fedot_ind.core.operation.filtration.feature_filtration import FeatureFilter

from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    *
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import \
    OneHotEncodingImplementation, LabelEncodingImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.decompose import \
    DecomposerClassImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_filters import \
    IsolationForestClassImplementation, IsolationForestRegImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_imbalanced_class import \
    ResampleImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_selectors import \
    NonLinearClassFSImplementation, LinearClassFSImplementation
from fedot_ind.core.models.nn.network_impl.explainable_convolution_model import XCModel
from fedot_ind.core.models.nn.network_impl.inception import InceptionTimeModel
from fedot_ind.core.models.nn.network_impl.omni_scale import OmniScaleModel
from fedot_ind.core.models.nn.network_impl.resnet import ResNetModel
from fedot_ind.core.models.nn.network_impl.tst import TSTModel

TEMPORARY_EXCLUDED = {
    'INDUSTRIAL_CLF_PREPROC_MODEL': {
        'rfe_lin_class': LinearClassFSImplementation,
        'rfe_non_lin_class': NonLinearClassFSImplementation,
    },
    'FEDOT_PREPROC_MODEL': {'pca': PCAImplementation,
                            'fast_ica': FastICAImplementation,
                            'poly_features': PolyFeaturesImplementation
                            },
    'INDUSTRIAL_PREPROC_MODEL': {'cat_features': DummyOperation,
                                 'dimension_reduction': FeatureFilter},
    'SKLEARN_REG_MODELS': {
        'gbr': GradientBoostingRegressor,
        'rfr': RandomForestRegressor,
        'adareg': AdaBoostRegressor,
        'linear': SklearnLinReg,
        # 'lgbmreg': LGBMRegressor,
    },
    'SKLEARN_CLF_MODELS': {'bernb': SklearnBernoulliNB,
                           'multinb': SklearnMultinomialNB,
                           # 'lgbm': LGBMClassifier
                           }}


class AtomizedModel(Enum):
    INDUSTRIAL_CLF_PREPROC_MODEL = {
        # for decomposed tasks
        'class_decompose': DecomposerClassImplementation,
        # for imbalanced data
        'resample': ResampleImplementation,

    }
    SKLEARN_CLF_MODELS = {
        # boosting models (bid datasets)
        'xgboost': XGBClassifier,
        # solo linear models
        'logit': SklearnLogReg,
        'knn': FedotKnnClassImplementation,
        # solo tree models
        'dt': DecisionTreeClassifier,
        # ensemble tree models
        'rf': RandomForestClassifier,
        # solo nn models
        'mlp': MLPClassifier
    }
    FEDOT_PREPROC_MODEL = {
        # data standartization
        'scaling': ScalingImplementation,
        'normalization': NormalizationImplementation,
        # missing data
        'simple_imputation': ImputationImplementation,
        # dimension reduction
        'kernel_pca': KernelPCAImplementation,
        # feature generation
        'topological_features': TopologicalFeaturesImplementation,
        # categorical encoding
        'one_hot_encoding': OneHotEncodingImplementation,
        'label_encoding': LabelEncodingImplementation
    }
    INDUSTRIAL_PREPROC_MODEL = {
        # data projection onto different basis
        'eigen_basis': EigenBasisImplementation,
        'wavelet_basis': WaveletBasisImplementation,
        'fourier_basis': FourierBasisImplementation,
        # feature extraction algorithm
        'topological_extractor': TopologicalExtractor,
        'quantile_extractor': QuantileExtractor,
        'signal_extractor': SignalExtractor,
        'recurrence_extractor': RecurrenceExtractor,
        # nn feature extraction algorithm
        'minirocket_extractor': MiniRocketExtractor,
        # isolation_forest forest
        'isolation_forest_class': IsolationForestClassImplementation,
        'isolation_forest_reg': IsolationForestRegImplementation,
    }

    SKLEARN_REG_MODELS = {
        # boosting models (bid datasets)
        'xgbreg': XGBRegressor,
        'sgdr': SklearnSGD,
        # ensemble tree models (big datasets)
        'treg': ExtraTreesRegressor,
        # solo linear models with regularization
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        # solo tree models (small datasets)
        'knnreg': FedotKnnRegImplementation,
        'dtreg': DecisionTreeRegressor
    }
    NEURAL_MODEL = {
        # fundamental models
        'inception_model': InceptionTimeModel,
        'omniscale_model': OmniScaleModel,
        'resnet_model': ResNetModel,
        # transformer models
        'tst_model': TSTModel,
        # explainable models
        'xcm_model': XCModel
    }

def default_industrial_availiable_operation(problem: str = 'regression'):
    operation_dict = {'regression': SKLEARN_REG_MODELS.keys(),
                      'forecastiong': SKLEARN_REG_MODELS.keys(),
                      'classification': SKLEARN_CLF_MODELS.keys()}
    available_operations = [operation_dict[problem],
                            NEURAL_MODEL.keys(),
                            INDUSTRIAL_PREPROC_MODEL.keys(),
                            FEDOT_PREPROC_MODEL.keys()]

    available_operations = list(chain(*[list(x) for x in available_operations]))
    excluded_operation = {'regression': ['one_hot_encoding',
                                         'label_encoding',
                                         'isolation_forest_class',
                                         'tst_model',
                                         'xcm_model',
                                         'resnet_model'],
                          'forecasting': ['one_hot_encoding',
                                          'label_encoding',
                                          'isolation_forest_class'],
                          'classification': [
                              'isolation_forest_reg',
                              'tst_model',
                              'xcm_model',
                              'one_hot_encoding',
                              'label_encoding',
                              'isolation_forest_class',
                          ]}
    available_operations = [x for x in available_operations if x not in excluded_operation[problem]]
    return available_operations

INDUSTRIAL_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_PREPROC_MODEL.value
INDUSTRIAL_CLF_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_CLF_PREPROC_MODEL.value
FEDOT_PREPROC_MODEL = AtomizedModel.FEDOT_PREPROC_MODEL.value
SKLEARN_CLF_MODELS = AtomizedModel.SKLEARN_CLF_MODELS.value
SKLEARN_REG_MODELS = AtomizedModel.SKLEARN_REG_MODELS.value
NEURAL_MODEL = AtomizedModel.NEURAL_MODEL.value

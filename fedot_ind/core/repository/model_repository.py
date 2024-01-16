from enum import Enum

from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import \
    LabelEncodingImplementation, OneHotEncodingImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.decompose import \
    DecomposerClassImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_filters import \
    IsolationForestClassImplementation, IsolationForestRegImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_imbalanced_class import \
    ResampleImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_selectors import \
    LinearClassFSImplementation, NonLinearClassFSImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    *
from fedot.core.operations.evaluation.operation_implementations.models.knn import FedotKnnClassImplementation, \
    FedotKnnRegImplementation
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.linear_model import (
    Lasso as SklearnLassoReg,
    LinearRegression as SklearnLinReg,
    LogisticRegression as SklearnLogReg,
    Ridge as SklearnRidgeReg,
    SGDRegressor as SklearnSGD
)
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB, MultinomialNB as SklearnMultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from fedot_ind.core.models.nn.network_impl.explainable_convolution_model import XCModel
from fedot_ind.core.models.nn.network_impl.inception import InceptionTimeModel
from fedot_ind.core.models.nn.network_impl.mini_rocket import MiniRocketExtractor
from fedot_ind.core.models.nn.network_impl.omni_scale import OmniScaleModel
from fedot_ind.core.models.nn.network_impl.resnet import ResNetModel
from fedot_ind.core.models.nn.network_impl.tst import TSTModel
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.recurrence.reccurence_extractor import RecurrenceExtractor
from fedot_ind.core.models.signal.signal_extractor import SignalExtractor
from fedot_ind.core.models.topological.topological_extractor import TopologicalExtractor
from fedot_ind.core.operation.dummy.dummy_operation import DummyOperation
from fedot_ind.core.operation.filtration.feature_filtration import FeatureFilter
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation


class TopologicalFeaturesImplementation:
    pass


TEMPORARY_EXCLUDED = {
    'INDUSTRIAL_CLF_PREPROC_MODEL': {'topological_features': TopologicalFeaturesImplementation},
    'FEDOT_PREPROC_MODEL': {'pca': PCAImplementation},
    'INDUSTRIAL_PREPROC_MODEL': {'cat_features': DummyOperation,
                                 'dimension_reduction': FeatureFilter}}


class AtomizedModel(Enum):
    INDUSTRIAL_CLF_PREPROC_MODEL = {
        'rfe_lin_class': LinearClassFSImplementation,
        'rfe_non_lin_class': NonLinearClassFSImplementation,
        'class_decompose': DecomposerClassImplementation,
        'resample': ResampleImplementation,
        'isolation_forest_class': IsolationForestClassImplementation,
        'isolation_forest_reg': IsolationForestRegImplementation,

    }
    SKLEARN_CLF_MODELS = {
        'xgboost': XGBClassifier,
        'knn': FedotKnnClassImplementation,
        'logit': SklearnLogReg,
        'bernb': SklearnBernoulliNB,
        'multinb': SklearnMultinomialNB,
        'dt': DecisionTreeClassifier,
        'rf': RandomForestClassifier,
        'mlp': MLPClassifier,
        # 'lgbm': LGBMClassifier

    }
    FEDOT_PREPROC_MODEL = {
        'scaling': ScalingImplementation,
        'normalization': NormalizationImplementation,
        'simple_imputation': ImputationImplementation,
        'kernel_pca': KernelPCAImplementation,
        'poly_features': PolyFeaturesImplementation,
        'one_hot_encoding': OneHotEncodingImplementation,
        'label_encoding': LabelEncodingImplementation,
        'fast_ica': FastICAImplementation
    }
    INDUSTRIAL_PREPROC_MODEL = {
        'eigen_basis': EigenBasisImplementation,
        'wavelet_basis': WaveletBasisImplementation,
        'fourier_basis': FourierBasisImplementation,
        'topological_extractor': TopologicalExtractor,
        'quantile_extractor': QuantileExtractor,
        'signal_extractor': SignalExtractor,
        'recurrence_extractor': RecurrenceExtractor,
        'minirocket_extractor': MiniRocketExtractor,

    }

    SKLEARN_REG_MODELS = {
        'xgbreg': XGBRegressor,
        'adareg': AdaBoostRegressor,
        'gbr': GradientBoostingRegressor,
        'dtreg': DecisionTreeRegressor,
        'treg': ExtraTreesRegressor,
        'rfr': RandomForestRegressor,
        'linear': SklearnLinReg,
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        'sgdr': SklearnSGD,
        # 'lgbmreg': LGBMRegressor,
        'knnreg': FedotKnnRegImplementation
    }
    NEURAL_MODEL = {
        'inception_model': InceptionTimeModel,
        'omniscale_model': OmniScaleModel,
        'tst_model': TSTModel,
        'resnet_model': ResNetModel,
        'xcm_model': XCModel
    }


INDUSTRIAL_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_PREPROC_MODEL.value
INDUSTRIAL_CLF_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_CLF_PREPROC_MODEL.value
FEDOT_PREPROC_MODEL = AtomizedModel.FEDOT_PREPROC_MODEL.value
SKLEARN_CLF_MODELS = AtomizedModel.SKLEARN_CLF_MODELS.value
SKLEARN_REG_MODELS = AtomizedModel.SKLEARN_REG_MODELS.value
NEURAL_MODEL = AtomizedModel.NEURAL_MODEL.value

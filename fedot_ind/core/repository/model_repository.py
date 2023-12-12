from enum import Enum

from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.topological_extractor import \
    TopologicalFeaturesImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.utilities.random import ImplementationRandomStateHandler
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, \
    RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
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
    IsolationForestClassImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_imbalanced_class import \
    ResampleImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_selectors import \
    NonLinearClassFSImplementation, LinearClassFSImplementation


class AtomizedModel(Enum):
    INDUSTRIAL_CLF_PREPROC_MODEL = {
        'rfe_lin_class': LinearClassFSImplementation,
        'rfe_non_lin_class': NonLinearClassFSImplementation,
        'class_decompose': DecomposerClassImplementation,
        'resample': ResampleImplementation,
        'isolation_forest_class': IsolationForestClassImplementation,
        'topological_features': TopologicalFeaturesImplementation
    }
    SKLEARN_CLF_MODELS = {
        'xgbreg': XGBRegressor,
        'adareg': AdaBoostRegressor,
        'gbr': GradientBoostingRegressor,
        'dtreg': DecisionTreeRegressor,
        'treg': ExtraTreesRegressor,
        'rfr': RandomForestRegressor,
        'linear': SklearnLinReg,
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        'lgbmreg': LGBMRegressor,
        'xgboost': XGBClassifier,
        'logit': SklearnLogReg,
        'rf': RandomForestClassifier,
        'mlp': MLPClassifier,
        'lgbm': LGBMClassifier,

    }
    FEDOT_PREPROC_MODEL = {
        'scaling': ScalingImplementation,
        'normalization': NormalizationImplementation,
        'simple_imputation': ImputationImplementation,
        'pca': PCAImplementation,
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
        'cat_features': DummyOperation,
        'dimension_reduction': FeatureFilter
    }


INDUSTRIAL_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_PREPROC_MODEL.value
INDUSTRIAL_CLF_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_CLF_PREPROC_MODEL.value
FEDOT_PREPROC_MODEL = AtomizedModel.FEDOT_PREPROC_MODEL.value
SKLEARN_CLF_MODELS = AtomizedModel.SKLEARN_CLF_MODELS.value

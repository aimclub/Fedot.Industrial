from enum import Enum
from itertools import chain

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

from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.fast_topological_extractor import \
    TopologicalFeaturesImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    ExogDataTransformationImplementation, GaussianFilterImplementation, LaggedTransformationImplementation, \
    SparseLaggedTransformationImplementation, TsSmoothingImplementation
from fedot.core.operations.evaluation.operation_implementations.models.knn import FedotKnnClassImplementation, \
    FedotKnnRegImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.arima import \
    STLForecastARIMAImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.cgru import \
    CGRUImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    AutoRegImplementation, ExpSmoothingImplementation, GLMImplementation
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestClassifier, RandomForestRegressor
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

from fedot_ind.core.models.manifold.riemann_embeding import RiemannExtractor
from fedot_ind.core.models.nn.network_impl.chronos_tst import ChronosExtractor
from fedot_ind.core.models.nn.network_impl.explainable_convolution_model import XCModel
from fedot_ind.core.models.nn.network_impl.inception import InceptionTimeModel
from fedot_ind.core.models.nn.network_impl.mini_rocket import MiniRocketExtractor
from fedot_ind.core.models.nn.network_impl.omni_scale import OmniScaleModel
from fedot_ind.core.models.nn.network_impl.resnet import ResNetModel
from fedot_ind.core.models.nn.network_impl.tst import TSTModel
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.recurrence.reccurence_extractor import RecurrenceExtractor
from fedot_ind.core.models.ts_forecasting.glm import GLMIndustrial
from fedot_ind.core.operation.dummy.dummy_operation import DummyOperation
from fedot_ind.core.operation.filtration.channel_filtration import ChannelCentroidFilter
from fedot_ind.core.operation.filtration.feature_filtration import FeatureFilter
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot_ind.core.repository.constanst_repository import EXCLUDED_OPERATION_MUTATION

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
    'NEURAL_MODELS': {'omniscale_model': OmniScaleModel,
                      # transformer models
                      'tst_model': TSTModel,
                      # explainable models
                      'xcm_model': XCModel
                      }
}


class AtomizedModel(Enum):
    INDUSTRIAL_CLF_PREPROC_MODEL = {
        # for decomposed tasks
        'class_decompose': DecomposerClassImplementation,
        # for imbalanced data
        'resample': ResampleImplementation,

    }
    SKLEARN_CLF_MODELS = {
        # boosting models (bid datasets)
        'xgboost': GradientBoostingClassifier,
        # solo linear models
        'logit': SklearnLogReg,
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
        'topological_extractor': TopologicalFeaturesImplementation
    }
    INDUSTRIAL_PREPROC_MODEL = {
        # data filtration
        'channel_filtration': ChannelCentroidFilter,
        # data projection onto different basis
        'eigen_basis': EigenBasisImplementation,
        'wavelet_basis': WaveletBasisImplementation,
        'fourier_basis': FourierBasisImplementation,
        # feature extraction algorithm
        'recurrence_extractor': RecurrenceExtractor,
        'quantile_extractor': QuantileExtractor,
        'riemann_extractor': RiemannExtractor,
        # feature generation
        'topological_extractor': TopologicalFeaturesImplementation,
        # nn feature extraction algorithm
        'minirocket_extractor': MiniRocketExtractor,
        # 'chronos_extractor': ChronosExtractor,
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
        'dtreg': DecisionTreeRegressor
    }

    FORECASTING_MODELS = {
        # boosting models (bid datasets)
        'ar': AutoRegImplementation,
        'stl_arima': STLForecastARIMAImplementation,
        'ets': ExpSmoothingImplementation,
        'cgru': CGRUImplementation,
        'glm': GLMIndustrial
    }

    FORECASTING_PREPROC = {
        'lagged': LaggedTransformationImplementation,
        'sparse_lagged': SparseLaggedTransformationImplementation,
        'smoothing': TsSmoothingImplementation,
        'gaussian_filter': GaussianFilterImplementation,
        'exog_ts': ExogDataTransformationImplementation,
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
                      'ts_forecasting': FORECASTING_MODELS.keys(),
                      'classification': SKLEARN_CLF_MODELS.keys()}

    if problem == 'ts_forecasting':
        available_operations = [operation_dict[problem],
                                FORECASTING_PREPROC.keys(),
                                SKLEARN_REG_MODELS.keys(),
                                INDUSTRIAL_PREPROC_MODEL.keys(),
                                ]
    else:
        available_operations = [operation_dict[problem],
                                NEURAL_MODEL.keys(),
                                INDUSTRIAL_PREPROC_MODEL.keys(),
                                FEDOT_PREPROC_MODEL.keys()]

    available_operations = list(
        chain(*[list(x) for x in available_operations]))
    excluded_operations = list(
        chain(*[list(TEMPORARY_EXCLUDED[x]) for x in TEMPORARY_EXCLUDED.keys()]))
    available_operations = [x for x in available_operations
                            if x not in EXCLUDED_OPERATION_MUTATION[problem] and x not in excluded_operations]
    return available_operations


INDUSTRIAL_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_PREPROC_MODEL.value
INDUSTRIAL_CLF_PREPROC_MODEL = AtomizedModel.INDUSTRIAL_CLF_PREPROC_MODEL.value
FEDOT_PREPROC_MODEL = AtomizedModel.FEDOT_PREPROC_MODEL.value
SKLEARN_CLF_MODELS = AtomizedModel.SKLEARN_CLF_MODELS.value
SKLEARN_REG_MODELS = AtomizedModel.SKLEARN_REG_MODELS.value
NEURAL_MODEL = AtomizedModel.NEURAL_MODEL.value
FORECASTING_MODELS = AtomizedModel.FORECASTING_MODELS.value
FORECASTING_PREPROC = AtomizedModel.FORECASTING_PREPROC.value

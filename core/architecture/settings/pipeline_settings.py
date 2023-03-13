from enum import Enum

from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from core.models.detection.probalistic.kalman import UnscentedKalmanFilter
from core.models.detection.subspaces.sst import SingularSpectrumTransformation
from core.operation.transformation.basis.chebyshev import ChebyshevBasis
from core.operation.transformation.basis.data_driven import DataDrivenBasisImplementation
from core.operation.transformation.basis.fourier import FourierBasis
from core.operation.transformation.basis.legendre import LegenderBasis
from core.operation.transformation.basis.power import PowerBasis
from core.models.detection.subspaces.func_pca import FunctionalPCA
from core.models.signal.RecurrenceExtractor import RecurrenceExtractor
from core.models.signal.SignalExtractor import SignalExtractor
from core.models.statistical.QuantileRunner import StatsExtractor
from core.models.topological.TopologicalRunner import TopologicalExtractor


class BasisTransformations(Enum):
    legender = LegenderBasis
    chebyshev = ChebyshevBasis
    datadriven = DataDrivenBasisImplementation
    power = PowerBasis
    Fourier = FourierBasis


class FeatureGenerator(Enum):
    statistical = StatsExtractor
    wavelet = SignalExtractor
    topological = TopologicalExtractor
    recurrence = RecurrenceExtractor


class MlModel(Enum):
    tsc = TimeSeriesClassifier
    functional_pca = FunctionalPCA
    kalman_filter = UnscentedKalmanFilter
    sst = SingularSpectrumTransformation

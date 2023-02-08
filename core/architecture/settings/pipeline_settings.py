from enum import Enum

from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from core.operation.transformation.basis.chebyshev import ChebyshevBasis
from core.operation.transformation.basis.data_driven import DataDrivenBasis
from core.operation.transformation.basis.fourier import FourierBasis
from core.operation.transformation.basis.legendre import LegenderBasis
from core.operation.transformation.basis.power import PowerBasis
from core.models.detection.subspaces.func_pca import FunctionalPCA
from core.models.signal.RecurrenceRunner import RecurrenceRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalRunner


class BasisTransformations(Enum):
    legender = LegenderBasis
    chebyshev = ChebyshevBasis
    datadriven = DataDrivenBasis
    power = PowerBasis
    Fourier = FourierBasis


class FeatureGenerator(Enum):
    statistical = StatsRunner
    wavelet = SignalRunner
    topological = TopologicalRunner
    recurrence = RecurrenceRunner


class MlModel(Enum):
    tsc = TimeSeriesClassifier
    functional_pca = FunctionalPCA

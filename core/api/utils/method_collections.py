from enum import Enum

from core.models.EnsembleRunner import EnsembleRunner
from core.models.signal.RecurrenceRunner import RecurrenceRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.spectral.SSARunner import SSARunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalRunner
from core.ensemble.baseline.AggEnsembler import AggregationEnsemble
from core.TimeSeriesClassifier import TimeSeriesClassifier


class FeatureGenerator(Enum):
    quantile = StatsRunner
    window_quantile = StatsRunner
    wavelet = SignalRunner
    spectral = SSARunner
    window_spectral = SSARunner
    topological = TopologicalRunner
    recurrence = RecurrenceRunner
    ensemble = EnsembleRunner


class WindowFeatureGenerator(Enum):
    window_quantile = StatsRunner
    window_spectral = SSARunner


class EnsembleGenerator(Enum):
    AGG_voting = AggregationEnsemble


class TaskGenerator(Enum):
    TSC = TimeSeriesClassifier

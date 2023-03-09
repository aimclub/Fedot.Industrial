from enum import Enum

from core.ensemble.baseline.AggEnsembler import AggregationEnsemble
from core.models.EnsembleRunner import EnsembleRunner
# from core.models.signal.RecurrenceRunner import RecurrenceRunner
from core.models.signal.SignalExtractor import SignalExtractor
from core.models.spectral.SSAExtractor import SSAExtractor
from core.models.statistical.QuantileRunner import StatsExtractor
# from core.models.topological.TopologicalRunner import TopologicalRunner
from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier, TimeSeriesImageClassifier


class FeatureGenerator(Enum):
    quantile = StatsExtractor
    window_quantile = StatsExtractor
    wavelet = SignalExtractor
    spectral = SSAExtractor
    window_spectral = SSAExtractor
    # topological = TopologicalRunner
    # recurrence = RecurrenceRunner
    ensemble = EnsembleRunner


class WindowFeatureGenerator(Enum):
    window_quantile = StatsExtractor
    window_spectral = SSAExtractor


class EnsembleGenerator(Enum):
    AGG_voting = AggregationEnsemble


class TaskGenerator(Enum):
    ts_classification = TimeSeriesClassifier
    ts_image_classification = TimeSeriesImageClassifier

"""Legacy benchmark wrappers kept inside the Industrial benchmark namespace."""

from benchmark.industrial.legacy.classification import BenchmarkTSC
from benchmark.industrial.legacy.forecasting import BenchmarkTSF
from benchmark.industrial.legacy.regression import BenchmarkTSER

__all__ = [
    "BenchmarkTSC",
    "BenchmarkTSER",
    "BenchmarkTSF",
]

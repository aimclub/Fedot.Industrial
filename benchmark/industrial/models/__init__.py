"""Benchmark model specs and built-in model adapter factories."""

from benchmark.industrial.models.classification import (
    FutureFusionClassifierAdapter,
    FutureMultimodalClassifierAdapter,
    KernelEnsembleClassifierAdapter,
    MajorityClassClassifier,
    MiniRocketRidgeClassifierAdapter,
    NearestCentroidClassifier,
    OptionalExternalClassifier,
    PDLClassifierAdapter,
    build_classification_model,
)
from benchmark.industrial.models.regression import (
    KernelEnsembleRegressorAdapter,
    LinearRegressor,
    MeanRegressor,
    OptionalExternalRegressor,
    PDLRegressorAdapter,
    build_regression_model,
)
from benchmark.industrial.models.specs import ModelSpec


def build_forecasting_model_adapter(*args, **kwargs):
    from benchmark.industrial.models.forecasting import build_forecasting_model_adapter as build_adapter

    return build_adapter(*args, **kwargs)


__all__ = [
    "FutureFusionClassifierAdapter",
    "FutureMultimodalClassifierAdapter",
    "KernelEnsembleClassifierAdapter",
    "KernelEnsembleRegressorAdapter",
    "LinearRegressor",
    "MajorityClassClassifier",
    "MeanRegressor",
    "MiniRocketRidgeClassifierAdapter",
    "ModelSpec",
    "NearestCentroidClassifier",
    "OptionalExternalClassifier",
    "OptionalExternalRegressor",
    "PDLClassifierAdapter",
    "PDLRegressorAdapter",
    "build_classification_model",
    "build_forecasting_model_adapter",
    "build_regression_model",
]

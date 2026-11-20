"""Explicit TSER controls and optional reference estimators from aeon."""

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from benchmark.industrial.core import RunStatus
from benchmark.industrial.errors import BenchmarkRegressionError


AEON_TYPES = {
    "freshprince_regressor": ("aeon.regression.feature_based", "FreshPRINCERegressor", ("tsfresh",)),
    "drcif_regressor": ("aeon.regression.interval_based", "DrCIFRegressor", ()),
    "dtw_1nn_regressor": ("aeon.regression.distance_based", "KNeighborsTimeSeriesRegressor", ()),
    "rocket_ridge_regressor": ("aeon.transformations.collection.convolution_based", "Rocket", ()),
}
TSER_ADAPTERS = frozenset(AEON_TYPES) | {"ed_1nn_regressor", "flat_forest_regressor"}


def restore_tser_panel(matrix, dimensions):
    """Recover the channel-major layout recorded by the dataset parser."""
    x = np.asarray(matrix, dtype=float)
    if (type(dimensions) is not int or dimensions <= 0 or x.ndim != 2 or not all(x.shape)
            or x.shape[1] % dimensions):
        raise BenchmarkRegressionError("TSER rows do not match the declared channel layout.")
    return x.reshape(len(x), dimensions, x.shape[1] // dimensions)


@dataclass
class TSERRegressor:
    adapter: str
    name: str
    params: dict
    tags: tuple[str, ...] = ("tser", "regression")
    estimator_: object = None

    @property
    def optional(self):
        return self.adapter in AEON_TYPES

    @property
    def uses_panel(self):
        return self.optional

    def _external_type(self):
        module, name, dependencies = AEON_TYPES[self.adapter]
        for dependency in dependencies:
            import_module(dependency)
        return getattr(import_module(module), name)

    def availability(self):
        if self.optional:
            try:
                self._external_type()
            except ImportError as error:
                return RunStatus.NOT_AVAILABLE, f"Install fedot-ind[tser]: {error}"
        return RunStatus.SUCCESS, "ready"

    def _estimator(self):
        params = dict(self.params)
        if self.adapter == "ed_1nn_regressor":
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(n_neighbors=1, metric="euclidean", **params)
        if self.adapter == "flat_forest_regressor":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**({"random_state": 0, "n_jobs": 1} | params))
        estimator = self._external_type()
        if self.adapter == "dtw_1nn_regressor":
            return estimator(**({"n_neighbors": 1, "distance": "dtw", "n_jobs": 1} | params))
        if self.adapter == "rocket_ridge_regressor":
            from sklearn.linear_model import RidgeCV
            from sklearn.pipeline import make_pipeline
            alphas = tuple(params.pop("alphas", (.1, 1., 10.)))
            return make_pipeline(estimator(**({"random_state": 0, "n_jobs": 1} | params)), RidgeCV(alphas=alphas))
        return estimator(**({"random_state": 0, "n_jobs": 1} | params))

    @staticmethod
    def _validate(features):
        x = np.asarray(features, dtype=float)
        if x.ndim not in (2, 3) or not all(x.shape) or not np.isfinite(x).all():
            raise BenchmarkRegressionError("TSER reference adapters require finite, equal-length inputs.")
        return x

    def fit(self, features, target):
        x, y = self._validate(features), np.asarray(target, dtype=float)
        if y.shape != (len(x),) or not np.isfinite(y).all():
            raise BenchmarkRegressionError("TSER requires one finite target per record.")
        self.estimator_ = self._estimator().fit(x, y)

    def predict(self, features):
        if self.estimator_ is None:
            raise BenchmarkRegressionError("Fit the TSER model before predicting.")
        x = self._validate(features)
        y = np.asarray(self.estimator_.predict(x), dtype=float)
        if y.shape != (len(x),) or not np.isfinite(y).all():
            raise BenchmarkRegressionError("TSER model did not return one finite value per record.")
        return y

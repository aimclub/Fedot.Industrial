from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.base import BaseEstimator

from fedot_ind.core.models.kernel.okhs_common import OKHSMethod, QPolicy
from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
from fedot_ind.core.models.kernel.rkbs import RKBSCompositeClassifier
from fedot_ind.core.operation.transformation.representation.kernel.kernels import OccupationKernel


@dataclass(frozen=True)
class ForecastCandidateScore:
    q: float
    score: float
    succeeded: bool
    error_message: str


class UnifiedOKHSAutoML(BaseEstimator):
    """
    Unified AutoML wrapper for OKHS/RKBS-based workflows across
    forecasting, classification, and regression tasks.
    """

    def __init__(
            self,
            task_type='auto',
            time_budget=300,
            q_range=(0.3, 0.9),
            q_grid_size=5,
            forecasting_method=OKHSMethod.DMD,
            q_policy=QPolicy.FIXED,
            verbose=False,
    ):
        self.task_type = task_type
        self.time_budget = time_budget
        self.q_range = q_range
        self.q_grid_size = q_grid_size
        self.forecasting_method = forecasting_method
        self.q_policy = q_policy
        self.verbose = verbose

        self._detect_task_type = task_type == 'auto'
        self.models_ = {}

    def fit(self, X, y=None, task_type=None):
        """Fit an OKHS/RKBS model for the inferred or specified task."""
        if task_type is None and self._detect_task_type:
            task_type = self._auto_detect_task_type(X, y)
        elif task_type is None:
            task_type = self.task_type

        self.task_type_ = task_type
        self.fit_summary_ = {
            'task_type': task_type,
            'time_budget': self.time_budget,
            'q_range': tuple(self.q_range),
            'q_grid_size': self.q_grid_size,
        }
        self.last_evaluation_error_ = None

        if task_type in ['classification', 'clf']:
            self._fit_classification(X, y)
        elif task_type in ['forecasting', 'forecast']:
            self._fit_forecasting(X)
        elif task_type in ['regression', 'reg']:
            self._fit_regression(X, y)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return self

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _auto_detect_task_type(self, X, y):
        """Infer task type from target availability and target cardinality."""
        if y is None:
            return 'forecasting'
        if len(np.unique(y)) / len(y) < 0.1:
            return 'classification'
        return 'regression'

    def _normalize_q_range(self) -> tuple[float, float]:
        q_min, q_max = self.q_range
        if not (0 < q_min <= q_max <= 1):
            raise ValueError(f"q_range must satisfy 0 < min <= max <= 1, got {self.q_range}")
        return float(q_min), float(q_max)

    def _build_q_grid(self) -> np.ndarray:
        q_min, q_max = self._normalize_q_range()
        if self.q_grid_size < 2:
            raise ValueError(f"q_grid_size must be at least 2, got {self.q_grid_size}")
        return np.linspace(q_min, q_max, self.q_grid_size)

    def _build_kernels(self) -> list[OccupationKernel]:
        return [
            OccupationKernel(q=float(q), kernel_type='rbf')
            for q in self._build_q_grid()
        ]

    def _build_forecaster(self, q: float) -> OKHSForecaster:
        return OKHSForecaster(
            q=float(q),
            method=self.forecasting_method,
            q_policy=self.q_policy,
        )

    def _fit_classification(self, X, y):
        """Fit the classification path with an explicit kernel grid."""
        kernels = self._build_kernels()
        self.kernel_q_values_ = [float(kernel.q) for kernel in kernels]
        self.model_ = RKBSCompositeClassifier(kernels=kernels, penalty='l1', verbose=self.verbose)
        self.model_.fit(X, y)

        self.fit_summary_.update({
            'penalty': 'l1',
            'kernel_count': len(kernels),
            'kernel_q_values': list(self.kernel_q_values_),
        })
        self._log(f"Fitted classification OKHS AutoML model with {len(kernels)} kernels")

    def _fit_forecasting(self, X):
        """Fit the forecasting path and store the q-selection report."""
        best_score = -np.inf
        best_q = 0.7
        candidate_scores: list[ForecastCandidateScore] = []

        for q in self._build_q_grid():
            model = self._build_forecaster(float(q))
            score = self._evaluate_forecaster(model, X)
            candidate_scores.append(
                ForecastCandidateScore(
                    q=float(q),
                    score=float(score),
                    succeeded=bool(np.isfinite(score)),
                    error_message='' if np.isfinite(score) else str(self.last_evaluation_error_ or ''),
                )
            )

            if score > best_score:
                best_score = score
                best_q = float(q)

        self.selected_q_ = best_q
        self.best_score_ = float(best_score)
        self.candidate_scores_ = [asdict(candidate) for candidate in candidate_scores]
        self.model_ = self._build_forecaster(best_q)
        self.model_.fit(X)

        self.fit_summary_.update({
            'selected_q': self.selected_q_,
            'best_score': self.best_score_,
            'candidate_scores': list(self.candidate_scores_),
            'forecasting_method': str(self.forecasting_method),
            'q_policy': str(self.q_policy),
        })
        self._log(f"Fitted forecasting OKHS AutoML model with q={best_q:.2f}")

    def _fit_regression(self, X, y):
        """Fit the regression path with the same kernel grid and L2 penalty."""
        kernels = self._build_kernels()
        self.kernel_q_values_ = [float(kernel.q) for kernel in kernels]
        self.model_ = RKBSCompositeClassifier(kernels=kernels, penalty='l2', verbose=self.verbose)
        self.model_.fit(X, y)

        self.fit_summary_.update({
            'penalty': 'l2',
            'kernel_count': len(kernels),
            'kernel_q_values': list(self.kernel_q_values_),
        })

    def _evaluate_forecaster(self, model, time_series):
        """Return a simple validation score for a forecasting candidate."""
        try:
            split_idx = len(time_series) * 3 // 4
            train_series = time_series[:split_idx]
            val_series = time_series[split_idx:]

            model.fit(train_series)
            predictions = np.asarray(model.predict())

            if len(predictions) == 0:
                self.last_evaluation_error_ = 'empty_prediction'
                return -np.inf

            self.last_evaluation_error_ = None
            return -float(np.mean(np.abs(predictions - val_series[:len(predictions)])))
        except Exception as exc:
            self.last_evaluation_error_ = exc
            return -np.inf

    def predict(self, X=None):
        """Dispatch prediction to the fitted task-specific model."""
        if not hasattr(self, 'model_'):
            raise ValueError('The model must be fitted before calling predict.')

        if self.task_type_ in ['classification', 'clf', 'regression', 'reg']:
            return self.model_.predict(X)
        if self.task_type_ in ['forecasting', 'forecast']:
            return self.model_.predict(X)
        raise ValueError(f"Unsupported task type for prediction: {self.task_type_}")

    def get_feature_importance(self):
        """Return kernel importance when the fitted estimator exposes it."""
        if hasattr(self.model_, 'kernel_importance_'):
            return self.model_.kernel_importance_
        return None


class OKHSEnhancedAutoML(UnifiedOKHSAutoML):
    """
    Extended AutoML variant with lightweight memory analysis used to narrow
    the fractional-order search range before delegating to the base class.
    """

    def __init__(
            self,
            task_type='auto',
            time_budget=300,
            enable_memory_analysis=True,
            ensemble_method='weighted',
            q_range=(0.3, 0.9),
            q_grid_size=5,
            forecasting_method=OKHSMethod.DMD,
            q_policy=QPolicy.FIXED,
            verbose=False,
    ):
        super().__init__(
            task_type=task_type,
            time_budget=time_budget,
            q_range=q_range,
            q_grid_size=q_grid_size,
            forecasting_method=forecasting_method,
            q_policy=q_policy,
            verbose=verbose,
        )
        self.enable_memory_analysis = enable_memory_analysis
        self.ensemble_method = ensemble_method

    def fit(self, X, y=None, task_type=None):
        """Fit with optional memory analysis and track elapsed time."""
        start_time = time.time()

        if self.enable_memory_analysis:
            self.memory_properties_ = self._analyze_memory_properties(X)
            self.optimal_q_ = self._select_optimal_q(self.memory_properties_)
        else:
            self.optimal_q_ = 0.7

        self.q_range = (
            max(0.1, self.optimal_q_ - 0.2),
            min(1.0, self.optimal_q_ + 0.2),
        )

        super().fit(X, y, task_type)

        self.training_time_ = time.time() - start_time
        self.fit_summary_['training_time'] = self.training_time_
        self.fit_summary_['optimal_q'] = self.optimal_q_
        return self

    def _analyze_memory_properties(self, data):
        """Analyze memory properties for a single series or a set of trajectories."""
        if isinstance(data, list) or (isinstance(data, np.ndarray) and data.ndim > 1):
            return self._analyze_trajectories_memory(data)
        return self._analyze_single_series_memory(data)

    def _analyze_single_series_memory(self, series):
        """Estimate the memory strength of a single time series via autocorrelation decay."""
        from statsmodels.tsa.stattools import acf

        autocorr = acf(series, nlags=min(len(series) - 1, 50), fft=True)
        lags = np.arange(1, len(autocorr))
        autocorr_vals = autocorr[1:]

        valid_mask = (autocorr_vals > 0) & (lags > 0)
        if np.sum(valid_mask) > 2:
            log_lags = np.log(lags[valid_mask])
            log_autocorr = np.log(autocorr_vals[valid_mask])
            slope, _ = np.polyfit(log_lags, log_autocorr, 1)

            if slope > -0.5:
                optimal_q = 0.9
            elif slope > -1.0:
                optimal_q = 0.7
            else:
                optimal_q = 0.3
        else:
            slope = -1.0
            optimal_q = 0.7

        return {
            'autocorrelation_slope': slope,
            'optimal_q': optimal_q,
            'memory_strength': 'strong' if optimal_q > 0.8 else 'medium' if optimal_q > 0.5 else 'weak',
        }

    def _analyze_trajectories_memory(self, trajectories):
        """Average memory diagnostics across a set of trajectories."""
        memory_properties = [
            self._analyze_single_series_memory(trajectory)
            for trajectory in trajectories
        ]

        avg_slope = np.mean([p['autocorrelation_slope'] for p in memory_properties])
        avg_q = np.mean([p['optimal_q'] for p in memory_properties])

        return {
            'autocorrelation_slope': avg_slope,
            'optimal_q': avg_q,
            'memory_strength': 'strong' if avg_q > 0.8 else 'medium' if avg_q > 0.5 else 'weak',
            'n_trajectories': len(trajectories),
        }

    def _select_optimal_q(self, memory_properties):
        """Extract the recommended q from memory diagnostics."""
        return memory_properties['optimal_q']

    def get_memory_report(self):
        """Return a small human-readable memory analysis report."""
        if hasattr(self, 'memory_properties_'):
            return (
                "TIME SERIES MEMORY ANALYSIS\n"
                "=============================\n"
                f"Memory strength: {self.memory_properties_['memory_strength']}\n"
                f"Optimal fractional order (q): {self.memory_properties_['optimal_q']:.2f}\n"
                f"Autocorrelation slope: {self.memory_properties_['autocorrelation_slope']:.3f}"
            )
        return "Memory analysis was not performed."

from __future__ import annotations

from dataclasses import dataclass

from fedot_ind.core.kernel_learning.contracts import KernelConfigValidationError


CLASSIFICATION_HEADS = ("rf", "logit", "xgboost", "catboost", "dt", "mlp", "lgbm")
REGRESSION_HEADS = ("treg", "ridge", "xgbreg", "dtreg", "lgbmreg", "catboostreg", "lasso")
FORECASTING_HEADS = (
    "ridge",
    "lagged_ridge_forecaster",
    "okhs_fdmd_forecaster",
    "topo_forecaster",
    "lagged_forecaster",
)
SAFE_PREPROCESSORS = ("scaling", "normalization", "simple_imputation", "kernel_pca")


@dataclass(frozen=True)
class KernelWarmStartTaskSpec:
    task_type: str
    default_head_model: str
    head_candidates: tuple[str, ...]


_TASK_REGISTRY = {
    "classification": KernelWarmStartTaskSpec(
        task_type="classification",
        default_head_model="rf",
        head_candidates=CLASSIFICATION_HEADS,
    ),
    "regression": KernelWarmStartTaskSpec(
        task_type="regression",
        default_head_model="treg",
        head_candidates=REGRESSION_HEADS,
    ),
    "forecasting": KernelWarmStartTaskSpec(
        task_type="forecasting",
        default_head_model="ridge",
        head_candidates=FORECASTING_HEADS,
    ),
    "ts_forecasting": KernelWarmStartTaskSpec(
        task_type="forecasting",
        default_head_model="ridge",
        head_candidates=FORECASTING_HEADS,
    ),
}


def resolve_warm_start_task(task_type: str) -> KernelWarmStartTaskSpec:
    key = str(task_type).strip().lower()
    spec = _TASK_REGISTRY.get(key)
    if spec is None:
        raise KernelConfigValidationError(f"Unsupported kernel warm-start task_type: {task_type}")
    return spec


def task_head_candidates(task_type: str) -> tuple[str, ...]:
    return resolve_warm_start_task(task_type).head_candidates


__all__ = [
    "CLASSIFICATION_HEADS",
    "FORECASTING_HEADS",
    "KernelWarmStartTaskSpec",
    "REGRESSION_HEADS",
    "SAFE_PREPROCESSORS",
    "resolve_warm_start_task",
    "task_head_candidates",
]

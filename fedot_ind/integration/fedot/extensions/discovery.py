"""Operation discovery derived from the Industrial catalog and active FEDOT."""

from __future__ import annotations

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum

from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog


_PROBLEM_TASKS = {
    "classification": TaskTypesEnum.classification,
    "classification_tabular": TaskTypesEnum.classification,
    "regression": TaskTypesEnum.regression,
    "regression_tabular": TaskTypesEnum.regression,
    "ts_forecasting": TaskTypesEnum.ts_forecasting,
    "anomaly_detection": TaskTypesEnum.classification,
}


def catalog_operation_names(problem: str) -> tuple[str, ...]:
    """Return stable operation names explicitly declared for a problem."""
    normalized = problem.removesuffix("_tabular")
    required_data_type = "tabular" if problem.endswith("_tabular") else None
    return tuple(operation.name for operation in load_industrial_extension_catalog().operations
                 if normalized in operation.problems
                 and (required_data_type is None or required_data_type in operation.data_types))


def default_industrial_available_operations(problem: str = "regression") -> list[str]:
    """Combine current FEDOT operations with Industrial catalog declarations."""
    try:
        task = _PROBLEM_TASKS[problem]
    except KeyError as error:
        raise ValueError(f"Unsupported Industrial problem: {problem!r}.") from error
    data_type = DataTypesEnum.tabular if problem.endswith("_tabular") else None
    base = []
    for repository_kind in ("model", "data_operation"):
        base.extend(OperationTypesRepository(repository_kind).suitable_operation(
            task_type=task,
            data_type=data_type,
        ))
    if problem == "anomaly_detection":
        anomaly_baselines = {"one_class_svm", "isolation_forest_class", "gaussian_filter"}
        base = [operation for operation in base if operation in anomaly_baselines]
    return sorted(set(base).union(catalog_operation_names(problem)))


# Preserve the historical misspelling for downstream callers.
default_industrial_availiable_operation = default_industrial_available_operations

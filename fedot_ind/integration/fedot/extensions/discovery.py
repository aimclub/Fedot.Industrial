"""Operation discovery derived from the Industrial catalog and active FEDOT."""

from __future__ import annotations

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum

from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
from fedot_ind.integration.fedot.extensions.operation_policy import excluded_operation_names


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
    excluded = excluded_operation_names(problem)
    return tuple(operation.name for operation in load_industrial_extension_catalog().operations
                 if normalized in operation.problems
                 and (required_data_type is None or required_data_type in operation.data_types)
                 and operation.name not in excluded)


def default_industrial_available_operations(problem: str = "regression") -> list[str]:
    """Combine current FEDOT operations with Industrial catalog declarations."""
    try:
        task = _PROBLEM_TASKS[problem]
    except KeyError as error:
        raise ValueError(f"Unsupported Industrial problem: {problem!r}.") from error
    data_type = _tabular_data_type() if problem.endswith("_tabular") else None
    base = []
    for repository_kind in ("model", "data_operation"):
        base.extend(OperationTypesRepository(repository_kind).suitable_operation(
            task_type=task,
            data_type=data_type,
        ))
    if problem == "anomaly_detection":
        anomaly_baselines = {"one_class_svm", "isolation_forest_class", "gaussian_filter"}
        base = [operation for operation in base if operation in anomaly_baselines]
    excluded = excluded_operation_names(problem)
    return sorted(set(base).union(catalog_operation_names(problem)).difference(excluded))


def _tabular_data_type():
    """Resolve the renamed table member across supported FEDOT profiles."""
    tabular = getattr(DataTypesEnum, "tabular", None)
    if tabular is not None:
        return tabular
    table = getattr(DataTypesEnum, "table", None)
    if table is not None:
        return table
    raise RuntimeError("The active FEDOT profile has no tabular data type.")


# Preserve the historical misspelling for downstream callers.
default_industrial_availiable_operation = default_industrial_available_operations

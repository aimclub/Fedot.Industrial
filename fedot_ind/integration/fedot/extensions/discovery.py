"""Operation discovery derived from the Industrial catalog and active FEDOT."""

from __future__ import annotations

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
                 and not operation.needs_explicit_task_type
                 and operation.name not in excluded)


def default_industrial_available_operations(problem: str = "regression") -> list[str]:
    """Return operations that satisfy the current TensorData extension contract."""
    try:
        _PROBLEM_TASKS[problem]
    except KeyError as error:
        raise ValueError(f"Unsupported Industrial problem: {problem!r}.") from error
    return sorted(catalog_operation_names(problem))


# Preserve the historical misspelling for downstream callers.
default_industrial_availiable_operation = default_industrial_available_operations

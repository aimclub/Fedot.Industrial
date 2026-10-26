"""Public index for explicit, operation-local tensor compute planning."""

from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend
from fedot_ind.core.optimizer.graph_validation.contracts import ComputeDevice

from .contracts import (
    ComputeRejection, ComputeRejectionCode, MemoryBudget, MemoryEstimate,
    TensorOperationCapabilities, TensorSpec,
)
from .planning import TensorOperationPlan, plan_tensor_operation
from .runtime import TensorExecutionResult, execute_tensor_operation

__all__ = [
    "ComputeDevice", "ComputeRejection", "ComputeRejectionCode", "EvaluationBackend",
    "MemoryBudget", "MemoryEstimate", "TensorExecutionResult", "TensorOperationCapabilities",
    "TensorOperationPlan", "TensorSpec", "execute_tensor_operation", "plan_tensor_operation",
]

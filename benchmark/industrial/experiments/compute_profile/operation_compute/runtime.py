"""Thin sequential interpreter; no model scheduling or global Torch mutation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Callable

from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend
from fedot_ind.core.optimizer.graph_validation.contracts import ComputeDevice

from .contracts import ComputeRejection, TensorSpec
from .planning import TensorOperationPlan, plan_tensor_operation

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class TensorExecutionResult:
    tensor: Tensor
    elapsed_seconds: float | None


def execute_tensor_operation(
        plan: TensorOperationPlan,
        tensor: Tensor,
        operation: Callable[[Tensor], Tensor],
        *,
        measure: bool = False,
) -> TensorExecutionResult:
    """Execute once; timing includes explicit transfer and operation completion."""
    if not isinstance(plan, TensorOperationPlan):
        raise TypeError("plan must be a TensorOperationPlan")
    if not isinstance(measure, bool):
        raise TypeError("measure must be boolean")
    admitted = plan_tensor_operation(
        plan.input_spec, plan.output_spec, plan.capabilities,
        device=plan.device, backend=plan.backend, budget=plan.budget,
    )
    if isinstance(admitted, ComputeRejection):
        raise ValueError(f"{admitted.code.value}: {admitted.message}")
    if plan.backend is not EvaluationBackend.SEQUENTIAL:
        raise ValueError("This operation-local interpreter executes only the sequential backend")
    if not callable(operation):
        raise TypeError("operation must be callable")

    import torch

    uses_cuda = ComputeDevice.CUDA in (plan.device, plan.input_spec.source_device)
    if uses_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable; no CPU fallback")
    cuda_index = torch.cuda.current_device() if uses_cuda else None
    _validate_tensor(tensor, plan.input_spec, "input", torch, cuda_index)
    if measure and uses_cuda:
        torch.cuda.synchronize(cuda_index)
    started = perf_counter() if measure else None
    argument = tensor
    if plan.input_spec.source_device is not plan.device:
        target = torch.device("cuda", cuda_index) if plan.device is ComputeDevice.CUDA else torch.device("cpu")
        argument = tensor.to(device=target)
    output = operation(argument)
    _validate_tensor(output, plan.output_spec, "output", torch, cuda_index)
    if measure and uses_cuda:
        torch.cuda.synchronize(cuda_index)
    elapsed = perf_counter() - started if measure else None
    return TensorExecutionResult(output, elapsed)


def _validate_tensor(tensor: Tensor, spec: TensorSpec, label: str, torch, cuda_index: int | None) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label} must be a Torch tensor")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(f"{label} must be a contiguous dense tensor for byte accounting")
    if tensor.requires_grad:
        raise ValueError(f"{label} must not require gradients in this operation-local interpreter")
    actual = (tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."),
              tensor.element_size(), tensor.device.type)
    expected = (spec.shape, spec.dtype, spec.element_size, spec.source_device.value)
    if actual != expected:
        raise ValueError(f"{label} tensor does not match TensorSpec: {actual} != {expected}")
    if spec.source_device is ComputeDevice.CUDA and tensor.device.index != cuda_index:
        raise ValueError(f"{label} CUDA tensor must use the current CUDA device")

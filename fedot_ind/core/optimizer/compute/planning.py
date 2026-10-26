"""Pure device, execution-backend admission and memory accounting."""

from __future__ import annotations

from dataclasses import dataclass

from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend
from fedot_ind.core.optimizer.graph_validation.contracts import ComputeDevice

from .contracts import (
    ComputeRejection, ComputeRejectionCode, MemoryBudget, MemoryEstimate,
    TensorOperationCapabilities, TensorSpec,
)


@dataclass(frozen=True)
class TensorOperationPlan:
    """Data only. Use plan_tensor_operation for admission; runtime rechecks it."""

    input_spec: TensorSpec
    output_spec: TensorSpec
    capabilities: TensorOperationCapabilities
    device: ComputeDevice
    backend: EvaluationBackend
    budget: MemoryBudget

    def __post_init__(self) -> None:
        for name, expected in (
            ("input_spec", TensorSpec), ("output_spec", TensorSpec),
            ("capabilities", TensorOperationCapabilities), ("device", ComputeDevice),
            ("backend", EvaluationBackend), ("budget", MemoryBudget),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be a {expected.__name__}")

    @property
    def memory(self) -> MemoryEstimate:
        # Keep the source alive through execution; count a separate destination
        # only on transfer. Output and estimated workspace coexist at the target.
        amounts = {ComputeDevice.CPU: 0, ComputeDevice.CUDA: 0}
        amounts[self.input_spec.source_device] += self.input_spec.nbytes
        transfer = (self.input_spec.nbytes
                    if self.input_spec.source_device is not self.device else 0)
        temporary = self.capabilities.estimated_temporary_bytes
        amounts[self.device] += transfer + self.output_spec.nbytes + temporary
        return MemoryEstimate(amounts[ComputeDevice.CPU], amounts[ComputeDevice.CUDA],
                              transfer, temporary)


def plan_tensor_operation(
        input_spec: TensorSpec,
        output_spec: TensorSpec,
        capabilities: TensorOperationCapabilities,
        *,
        device: ComputeDevice,
        backend: EvaluationBackend,
        budget: MemoryBudget,
) -> TensorOperationPlan | ComputeRejection:
    """No probing, auto selection, fallback, clients, tensors or Torch imports."""
    plan = TensorOperationPlan(input_spec, output_spec, capabilities, device, backend, budget)
    code = ComputeRejectionCode
    if backend is EvaluationBackend.LEGACY:
        return ComputeRejection(code.BACKEND_NOT_SUPPORTED, "An explicit execution backend is required")
    if (ComputeDevice.CUDA in (device, input_spec.source_device)
            and backend is not EvaluationBackend.SEQUENTIAL):
        return ComputeRejection(code.CUDA_BACKEND_NOT_ADMITTED,
                                "CUDA requires sequential operation-local execution")
    if device not in capabilities.devices:
        return ComputeRejection(code.DEVICE_NOT_SUPPORTED, "Operation does not declare the target device")
    if any(spec.dtype not in capabilities.dtypes for spec in (input_spec, output_spec)):
        return ComputeRejection(code.DTYPE_NOT_SUPPORTED, "Operation does not declare the tensor dtype")
    if output_spec.source_device is not device:
        return ComputeRejection(code.OUTPUT_DEVICE_MISMATCH, "Output must remain on the explicit target device")
    memory = plan.memory
    if memory.cpu_bytes > budget.cpu_bytes or memory.cuda_bytes > budget.cuda_bytes:
        return ComputeRejection(code.MEMORY_BUDGET_EXCEEDED,
                                "Estimated operation-local memory exceeds the declared budget")
    return plan

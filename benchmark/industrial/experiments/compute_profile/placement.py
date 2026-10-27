"""Public MTF invocation through an explicit device/backend/memory plan."""
from dataclasses import asdict

import torch

from fedot_ind.core.operation.transformation.torch_backend.image.mtf_transformation import MTF
from benchmark.industrial.experiments.compute_profile.operation_compute import (
    ComputeRejection, MemoryBudget, TensorOperationCapabilities, TensorSpec,
    execute_tensor_operation, plan_tensor_operation,
)
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend
from fedot_ind.core.optimizer.graph_validation.contracts import ComputeDevice


def probe_mtf_placement(settings, device):
    case = next(case for case in settings["cases"] if case["operation"] == "mtf")
    batch, length, bins, size = (case[k] for k in ("batch", "length", "bins", "image_size"))
    target = ComputeDevice(device)
    generator = torch.Generator().manual_seed(settings["seed"])
    source = torch.randn((batch, length), generator=generator)
    # Conservative workload accounting, not a hard allocator or process-RSS cap.
    # Counts and normalized transition matrices coexist. Include quadratic
    # buffers even for a two-point series with a large declared alphabet.
    workspace = batch * (length * bins * 32 + length * 80 + size * size * 16 + bins * bins * 16)
    if size == length or length <= bins:
        workspace += batch * length * length * 32
    capabilities = TensorOperationCapabilities(
        "MTF", (ComputeDevice.CPU, ComputeDevice.CUDA), ("float32",), workspace,
    )
    budget = settings["memory_budget_bytes"]
    plan = plan_tensor_operation(
        TensorSpec((batch, length), "float32", 4, ComputeDevice.CPU),
        TensorSpec((batch, size, size), "float32", 4, target), capabilities,
        device=target, backend=EvaluationBackend.SEQUENTIAL,
        budget=MemoryBudget(cpu_bytes=budget, cuda_bytes=budget),
    )
    if isinstance(plan, ComputeRejection):
        raise ValueError(f"MTF placement rejected: {plan.code.value}: {plan.message}")
    transform = MTF({"image_size": size, "n_bins": bins, "torch_device": device})
    result = execute_tensor_operation(plan, source, transform.transform, measure=True)
    reference = MTF({"image_size": size, "n_bins": bins, "torch_device": "cpu"}).transform(source)
    torch.testing.assert_close(result.tensor.cpu(), reference, rtol=settings["rtol"], atol=settings["atol"])
    return {"operation": "MTF", "device": device, "backend": plan.backend.value,
            "memory_estimate": asdict(plan.memory), "estimated_budget_bytes_per_device": budget,
            "transfer_and_operation_seconds": result.elapsed_seconds,
            "output_shape": list(result.tensor.shape), "within_tolerance": True,
            "scope": "One sequential MTF call; no whole-model CUDA or distributed GPU admission"}

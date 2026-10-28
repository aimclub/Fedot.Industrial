# Operation-local tensor compute

Public imports live in `fedot_ind.core.optimizer.compute`:

- `TensorSpec(shape, dtype, element_size, source_device)` describes a contiguous,
  dense, gradient-free tensor. Scalar shape `()` has one element; zero axes have
  zero bytes. Canonical dtype names include `float32`, not `torch.float32`.
- `TensorOperationCapabilities(name, devices, dtypes, estimated_temporary_bytes)`
  declares one operation, not model or Industrial-wide CUDA support.
- `plan_tensor_operation(input_spec, output_spec, capabilities, *, device,
  backend, budget)` returns `TensorOperationPlan | ComputeRejection`.
- `MemoryBudget(cpu_bytes, cuda_bytes=0)` bounds the estimate, not the allocator.
- `execute_tensor_operation(plan, tensor, operation, *, measure=False)` returns
  `TensorExecutionResult(tensor, elapsed_seconds)` and invokes the callable once.

`ComputeDevice` and `EvaluationBackend` reuse existing contracts. Both choices
are explicit: there is no auto mode or fallback. CUDA involvement, including a
CUDA source transferred to CPU, requires `SEQUENTIAL`. CUDA with `PROCESS` or
`DASK` is rejected pending verified admission. CPU plans may describe either
backend, but this interpreter only executes `SEQUENTIAL`; it never starts workers
or clients. `LEGACY` is not an explicit admitted backend.

The output spec must name the target device. The estimate retains source input,
adds destination input only on transfer, and adds output plus declared temporary
bytes at the target. `plan.memory` exposes CPU/CUDA totals, transfer bytes and
temporary bytes. This is not a hard peak-memory guarantee: allocator caches,
runtime workspaces, shared backing storage, callback state and other concurrent
allocations are outside the estimate. The caller owns the workspace estimate and
must provide an operation without hidden device transfers or global mutations.

Runtime validates input/output shape, dtype, element size, device, dense layout,
contiguity and absence of autograd. CUDA uses the current device only; multi-GPU
placement is not admitted. Unavailable CUDA fails before transfer/callback. A
cross-device input uses explicit `.to(device=...)`; output stays on target.
Only `measure=True` synchronizes CUDA and measures transfer plus execution.
No Torch settings, seeds, `EvolutionConfig`, model execution, or thresholds change.
Plans contain no tensors, callables or live clients.

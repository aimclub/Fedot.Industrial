from dataclasses import replace

import pytest
import torch

from fedot_ind.core.optimizer.compute import (
    ComputeDevice, EvaluationBackend, MemoryBudget, TensorOperationCapabilities,
    TensorOperationPlan, TensorSpec, execute_tensor_operation, plan_tensor_operation,
)
from fedot_ind.core.optimizer.compute import runtime


CPU = ComputeDevice.CPU
CUDA = ComputeDevice.CUDA


def make_plan(source=CPU, target=CPU, **overrides):
    arguments = dict(
        input_spec=TensorSpec((2, 3), "float32", 4, source),
        output_spec=TensorSpec((2, 3), "float32", 4, target),
        capabilities=TensorOperationCapabilities("negate", (CPU, CUDA)), device=target,
        backend=EvaluationBackend.SEQUENTIAL, budget=MemoryBudget(1000, 1000),
    )
    arguments.update(overrides)
    plan = plan_tensor_operation(**arguments)
    assert isinstance(plan, TensorOperationPlan)
    return plan


@pytest.mark.parametrize("measure", [False, True])
def test_real_cpu_operation_runs_once_without_cuda_probing_or_global_changes(monkeypatch, measure):
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    calls = []
    rng = torch.get_rng_state().clone()
    settings = (torch.get_num_threads(), torch.get_default_dtype(), torch.are_deterministic_algorithms_enabled())

    def forbidden(*args):
        pytest.fail("CPU execution must not probe or synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)

    def operation(value):
        calls.append(value)
        return torch.neg(value)

    result = execute_tensor_operation(make_plan(), tensor, operation, measure=measure)
    assert len(calls) == 1 and calls[0] is tensor
    assert torch.equal(result.tensor, -tensor)
    assert (result.elapsed_seconds is not None) is measure
    if measure:
        assert result.elapsed_seconds >= 0
    assert torch.equal(torch.get_rng_state(), rng)
    assert settings == (torch.get_num_threads(), torch.get_default_dtype(),
                        torch.are_deterministic_algorithms_enabled())


@pytest.mark.parametrize("tensor", [
    torch.ones(3, 2), torch.ones(2, 3, dtype=torch.float64),
    torch.ones(2, 3, requires_grad=True), torch.ones(3, 2).T,
    torch.ones(2, 3).to_sparse(), torch.empty(2, 3, device="meta"),
])
def test_actual_input_mismatch_short_circuits_callback(tensor):
    calls = []
    with pytest.raises(ValueError):
        execute_tensor_operation(make_plan(), tensor, lambda value: calls.append(value))
    assert calls == []


@pytest.mark.parametrize("output", [
    None, torch.ones(3, 2), torch.ones(2, 3, dtype=torch.int64),
    torch.ones(2, 3, requires_grad=True), torch.ones(3, 2).T,
])
def test_runtime_rejects_callback_output_outside_spec(output):
    with pytest.raises((TypeError, ValueError)):
        execute_tensor_operation(make_plan(), torch.ones(2, 3), lambda value: output)


def test_unavailable_cuda_fails_without_callback_or_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        execute_tensor_operation(make_plan(target=CUDA), torch.ones(2, 3), lambda value: calls.append(value))
    assert calls == []


def test_scalar_output_and_declared_dtype_conversion_use_their_own_specs():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    plan = make_plan(
        output_spec=TensorSpec((), "float64", 8, CPU),
        capabilities=TensorOperationCapabilities("sum", dtypes=("float32", "float64")),
        budget=MemoryBudget(32),
    )
    assert plan.memory.cpu_bytes == 32
    result = execute_tensor_operation(plan, tensor, lambda value: value.sum(dtype=torch.float64))
    assert result.tensor.shape == ()
    assert result.tensor.dtype == torch.float64
    assert result.tensor.item() == 15


@pytest.mark.parametrize("backend", [EvaluationBackend.PROCESS, EvaluationBackend.DASK])
def test_runtime_does_not_silently_execute_nonsequential_plans(backend):
    calls = []
    with pytest.raises(ValueError, match="only the sequential backend"):
        execute_tensor_operation(make_plan(backend=backend), torch.ones(2, 3), lambda value: calls.append(value))
    assert calls == []


@pytest.mark.parametrize("change,reason", [
    ({"budget": MemoryBudget(0)}, "memory_budget_exceeded"),
    ({"backend": EvaluationBackend.LEGACY}, "backend_not_supported"),
])
def test_runtime_readmits_directly_constructed_or_replaced_plans(change, reason):
    calls = []
    with pytest.raises(ValueError, match=reason):
        execute_tensor_operation(replace(make_plan(), **change), torch.ones(2, 3), lambda value: calls.append(value))
    assert calls == []


@pytest.mark.parametrize("backend", [EvaluationBackend.PROCESS, EvaluationBackend.DASK])
def test_forged_cuda_backend_plan_is_rejected_before_probing(monkeypatch, backend):
    plan = replace(make_plan(target=CUDA), backend=backend)

    def forbidden(*args):
        pytest.fail("Unadmitted CUDA execution must have no runtime effects")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(ValueError, match="cuda_backend_not_admitted"):
        execute_tensor_operation(plan, torch.ones(2, 3), forbidden)


def install_fake_cuda(monkeypatch, events):
    class FakeTensor:
        shape = (2, 3)
        dtype = torch.float32
        layout = torch.strided
        requires_grad = False

        def __init__(self, device):
            self.device = torch.device(device)

        def is_contiguous(self):
            return True

        def element_size(self):
            return 4

        def to(self, *, device):
            events.append(("to", str(device)))
            return FakeTensor(device)

    monkeypatch.setattr(torch, "Tensor", FakeTensor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index: events.append(("sync", index)))
    ticks = iter((10.0, 10.5))
    monkeypatch.setattr(runtime, "perf_counter", lambda: next(ticks))
    return FakeTensor


@pytest.mark.parametrize("source,target", [(CPU, CUDA), (CUDA, CPU), (CUDA, CUDA)])
@pytest.mark.parametrize("measure", [False, True])
def test_cuda_transfer_is_explicit_and_sync_occurs_only_during_measurement(monkeypatch, source, target, measure):
    events = []
    fake = install_fake_cuda(monkeypatch, events)
    source_name = "cpu" if source is CPU else "cuda:0"
    target_name = "cpu" if target is CPU else "cuda:0"

    def operation(value):
        events.append(("operation", str(value.device)))
        return fake(value.device)

    result = execute_tensor_operation(make_plan(source, target), fake(source_name), operation, measure=measure)
    expected = [("to", target_name)] if source is not target else []
    expected += [("operation", target_name)]
    if measure:
        expected = [("sync", 0)] + expected + [("sync", 0)]
    assert events == expected
    assert str(result.tensor.device) == target_name
    assert result.elapsed_seconds == (0.5 if measure else None)


def test_noncurrent_cuda_source_is_rejected_before_transfer(monkeypatch):
    events = []
    fake = install_fake_cuda(monkeypatch, events)
    with pytest.raises(ValueError, match="current CUDA device"):
        execute_tensor_operation(make_plan(CUDA, CPU), fake("cuda:1"), lambda value: events.append("operation"))
    assert events == []


def test_cuda_output_device_is_validated(monkeypatch):
    events = []
    fake = install_fake_cuda(monkeypatch, events)
    with pytest.raises(ValueError, match="output.*TensorSpec"):
        execute_tensor_operation(make_plan(target=CUDA), fake("cpu"), lambda value: fake("cpu"))


def test_cuda_output_ordinal_is_validated(monkeypatch):
    events = []
    fake = install_fake_cuda(monkeypatch, events)
    with pytest.raises(ValueError, match="output CUDA tensor must use the current CUDA device"):
        execute_tensor_operation(make_plan(CUDA, CUDA), fake("cuda:0"), lambda value: fake("cuda:1"))


def test_actual_element_size_is_validated_before_callback(monkeypatch):
    events = []
    fake = install_fake_cuda(monkeypatch, events)
    tensor = fake("cpu")
    tensor.element_size = lambda: 8
    with pytest.raises(ValueError, match="input.*TensorSpec"):
        execute_tensor_operation(make_plan(), tensor, lambda value: events.append("operation"))
    assert events == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires real CUDA hardware and a CUDA Torch build")
def test_real_cuda_operation_local_smoke():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = execute_tensor_operation(make_plan(target=CUDA), tensor, torch.neg, measure=True)
    assert torch.equal(result.tensor.cpu(), -tensor)
    assert result.elapsed_seconds >= 0

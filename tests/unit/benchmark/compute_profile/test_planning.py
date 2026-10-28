from dataclasses import FrozenInstanceError, asdict, replace
from itertools import product
import pickle
import subprocess
import sys

import pytest
from hypothesis import given, strategies as st

from benchmark.industrial.experiments.compute_profile.operation_compute import (
    ComputeDevice, ComputeRejection, ComputeRejectionCode, EvaluationBackend,
    MemoryBudget, TensorOperationCapabilities, TensorOperationPlan, TensorSpec,
    plan_tensor_operation,
)


CPU = ComputeDevice.CPU
CUDA = ComputeDevice.CUDA
SHAPES = st.lists(st.integers(0, 6), max_size=3).map(tuple)


def request(**overrides):
    arguments = dict(
        input_spec=TensorSpec((2, 3), "float32", 4, CPU),
        output_spec=TensorSpec((2, 3), "float32", 4, CPU),
        capabilities=TensorOperationCapabilities("negate"),
        device=CPU, backend=EvaluationBackend.SEQUENTIAL, budget=MemoryBudget(1000, 1000),
    )
    arguments.update(overrides)
    return plan_tensor_operation(**arguments)


@given(SHAPES, st.sampled_from([("float32", 4), ("bool", 1), ("complex128", 16)]))
def test_nbytes_matches_enumerated_coordinates_including_scalars_and_empty_axes(shape, dtype_size):
    dtype, size = dtype_size
    spec = TensorSpec(shape, dtype, size, CPU)
    coordinates = list(product(*(range(dimension) for dimension in shape)))
    assert spec.nbytes == len(coordinates) * size


@pytest.mark.parametrize("source,target", list(product((CPU, CUDA), repeat=2)))
@given(shape=SHAPES, output_shape=SHAPES, temporary=st.integers(0, 1000))
def test_memory_conserves_source_transfer_output_and_workspace(source, target, shape, output_shape, temporary):
    source_spec = TensorSpec(shape, "float32", 4, source)
    output_spec = replace(source_spec, shape=output_shape, source_device=target)
    caps = TensorOperationCapabilities("negate", (CPU, CUDA), estimated_temporary_bytes=temporary)
    plan = request(input_spec=source_spec, output_spec=output_spec, capabilities=caps,
                   device=target, budget=MemoryBudget(100000, 100000))
    assert isinstance(plan, TensorOperationPlan)
    expected_transfer = source_spec.nbytes if source != target else 0
    assert plan.memory.transfer_bytes == expected_transfer
    assert plan.memory.temporary_bytes == temporary
    assert plan.memory.cpu_bytes + plan.memory.cuda_bytes == (
        source_spec.nbytes + expected_transfer + output_spec.nbytes + temporary)
    source_only = source_spec.nbytes if source != target else 0
    totals = {CPU: plan.memory.cpu_bytes, CUDA: plan.memory.cuda_bytes}
    assert totals[target] == source_spec.nbytes + output_spec.nbytes + temporary
    assert totals[CUDA if target == CPU else CPU] == source_only


@given(temporary=st.integers(0, 10000), extra=st.integers(0, 10000))
def test_budget_boundary_and_monotonicity(temporary, extra):
    caps = TensorOperationCapabilities("negate", estimated_temporary_bytes=temporary)
    required = 48 + temporary
    accepted = request(capabilities=caps, budget=MemoryBudget(required))
    larger = request(capabilities=caps, budget=MemoryBudget(required + extra))
    rejected = request(capabilities=caps, budget=MemoryBudget(required - 1))
    assert isinstance(accepted, TensorOperationPlan)
    assert isinstance(larger, TensorOperationPlan)
    assert accepted.memory == larger.memory
    assert rejected.code is ComputeRejectionCode.MEMORY_BUDGET_EXCEEDED


@pytest.mark.parametrize("backend", [EvaluationBackend.PROCESS, EvaluationBackend.DASK])
@pytest.mark.parametrize("source,target", [(CPU, CUDA), (CUDA, CPU), (CUDA, CUDA)])
def test_cuda_never_enters_process_or_dask_without_admission(backend, source, target):
    result = request(
        input_spec=TensorSpec((2, 3), "float32", 4, source),
        output_spec=TensorSpec((2, 3), "float32", 4, target),
        capabilities=TensorOperationCapabilities("negate", (CPU, CUDA)), device=target, backend=backend,
    )
    assert result.code is ComputeRejectionCode.CUDA_BACKEND_NOT_ADMITTED


@pytest.mark.parametrize("backend", [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS, EvaluationBackend.DASK])
def test_cpu_admission_matches_the_available_interpreter(backend):
    plan = request(backend=backend)
    if backend is EvaluationBackend.SEQUENTIAL:
        assert isinstance(plan, TensorOperationPlan)
        assert plan.backend is backend
        assert plan.device is CPU
    else:
        assert isinstance(plan, ComputeRejection)
        assert plan.code is ComputeRejectionCode.BACKEND_NOT_SUPPORTED


@pytest.mark.parametrize("change,code", [
    ({"backend": EvaluationBackend.LEGACY}, ComputeRejectionCode.BACKEND_NOT_SUPPORTED),
    ({"device": CUDA}, ComputeRejectionCode.DEVICE_NOT_SUPPORTED),
    ({"output_spec": TensorSpec((2, 3), "float64", 8, CPU)}, ComputeRejectionCode.DTYPE_NOT_SUPPORTED),
    ({"output_spec": TensorSpec((2, 3), "float32", 4, CUDA)}, ComputeRejectionCode.OUTPUT_DEVICE_MISMATCH),
    ({"budget": MemoryBudget(47)}, ComputeRejectionCode.MEMORY_BUDGET_EXCEEDED),
])
def test_expected_rejections_are_typed(change, code):
    result = request(**change)
    assert isinstance(result, ComputeRejection)
    assert result.code is code


def test_transfer_requires_budgets_for_both_source_and_destination():
    changes = dict(
        output_spec=TensorSpec((2, 3), "float32", 4, CUDA), device=CUDA,
        capabilities=TensorOperationCapabilities("negate", (CPU, CUDA), estimated_temporary_bytes=12),
    )
    plan = request(**changes, budget=MemoryBudget(24, 60))
    assert isinstance(plan, TensorOperationPlan)
    assert plan.memory.cpu_bytes == 24
    assert plan.memory.cuda_bytes == 60
    for budget in (MemoryBudget(23, 60), MemoryBudget(24, 59)):
        assert request(**changes, budget=budget).code is ComputeRejectionCode.MEMORY_BUDGET_EXCEEDED


def test_plan_is_immutable_pickleable_and_contains_only_descriptive_data():
    plan = request()
    assert isinstance(plan, TensorOperationPlan)
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert request() == plan
    assert asdict(plan)["input_spec"]["shape"] == (2, 3)
    with pytest.raises(FrozenInstanceError):
        plan.device = CUDA


def test_planning_import_does_not_load_tensor_or_distributed_runtimes():
    completed = subprocess.run([
        sys.executable, "-c",
        "import sys; import benchmark.industrial.experiments.compute_profile.operation_compute; "
        "assert 'torch' not in sys.modules; assert 'distributed' not in sys.modules",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("changes", [
    {"shape": [2, 3]}, {"shape": (-1,)}, {"shape": (True,)}, {"shape": (1.5,)},
    {"dtype": "torch.float32"}, {"dtype": []}, {"element_size": 8},
    {"element_size": False}, {"source_device": "auto"}, {"source_device": "cpu"},
])
def test_tensor_spec_rejects_untyped_or_inconsistent_values(changes):
    arguments = dict(shape=(2, 3), dtype="float32", element_size=4, source_device=CPU)
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError)):
        TensorSpec(**arguments)


@pytest.mark.parametrize("changes", [
    {"devices": ()}, {"devices": [CPU]}, {"devices": (CPU, CPU)}, {"devices": ("cuda",)},
    {"dtypes": ()}, {"dtypes": ("invalid",)}, {"dtypes": ("float32", "float32")},
    {"estimated_temporary_bytes": -1}, {"estimated_temporary_bytes": True}, {"name": ""},
])
def test_capabilities_reject_invalid_declarations(changes):
    with pytest.raises((TypeError, ValueError)):
        TensorOperationCapabilities(**(dict(name="negate") | changes))


@pytest.mark.parametrize("value", [-1, True, 1.5, "100", None])
def test_budget_rejects_invalid_byte_counts(value):
    with pytest.raises(ValueError):
        MemoryBudget(value)
    with pytest.raises(ValueError):
        MemoryBudget(100, value)


@pytest.mark.parametrize("change", [{"device": "cpu"}, {"backend": "sequential"}, {"budget": 1000}])
def test_plan_requires_typed_choices_no_auto_or_string_flags(change):
    with pytest.raises(TypeError):
        request(**change)

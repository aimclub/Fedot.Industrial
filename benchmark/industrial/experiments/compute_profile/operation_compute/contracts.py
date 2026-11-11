"""Tensor-free contracts for one operation, not whole-model CUDA support."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from types import MappingProxyType

from fedot_ind.core.optimizer.graph_validation.contracts import ComputeDevice


DTYPE_ELEMENT_SIZES = MappingProxyType({
    "bool": 1, "uint8": 1, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
    "complex64": 8, "complex128": 16,
})


def nonnegative_bytes(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    element_size: int
    source_device: ComputeDevice

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            raise TypeError("shape must be a tuple")
        for dimension in self.shape:
            nonnegative_bytes(dimension, "shape dimension")
        if not isinstance(self.dtype, str) or self.dtype not in DTYPE_ELEMENT_SIZES:
            raise ValueError("dtype must be a supported canonical Torch dtype name")
        nonnegative_bytes(self.element_size, "element_size")
        if self.element_size != DTYPE_ELEMENT_SIZES[self.dtype]:
            raise ValueError("element_size does not match dtype")
        if not isinstance(self.source_device, ComputeDevice):
            raise TypeError("source_device must be a ComputeDevice")

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.element_size


@dataclass(frozen=True)
class TensorOperationCapabilities:
    """Caller-declared scope and workspace estimate for a single callable."""

    name: str
    devices: tuple[ComputeDevice, ...] = (ComputeDevice.CPU,)
    dtypes: tuple[str, ...] = ("float32",)
    estimated_temporary_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Operation name must not be empty")
        if (not isinstance(self.devices, tuple) or not self.devices
                or any(not isinstance(value, ComputeDevice) for value in self.devices)
                or len(set(self.devices)) != len(self.devices)):
            raise ValueError("devices must be a non-empty tuple of unique ComputeDevice values")
        if (not isinstance(self.dtypes, tuple) or not self.dtypes
                or any(not isinstance(value, str) or value not in DTYPE_ELEMENT_SIZES
                       for value in self.dtypes)
                or len(set(self.dtypes)) != len(self.dtypes)):
            raise ValueError("dtypes must be a non-empty tuple of unique supported names")
        nonnegative_bytes(self.estimated_temporary_bytes, "estimated_temporary_bytes")


@dataclass(frozen=True)
class MemoryBudget:
    """Operation-local accounting limits, not free RAM/VRAM or an allocator cap."""

    cpu_bytes: int
    cuda_bytes: int = 0

    def __post_init__(self) -> None:
        nonnegative_bytes(self.cpu_bytes, "cpu_bytes")
        nonnegative_bytes(self.cuda_bytes, "cuda_bytes")


@dataclass(frozen=True)
class MemoryEstimate:
    cpu_bytes: int
    cuda_bytes: int
    transfer_bytes: int
    temporary_bytes: int


class ComputeRejectionCode(str, Enum):
    DEVICE_NOT_SUPPORTED = "device_not_supported"
    DTYPE_NOT_SUPPORTED = "dtype_not_supported"
    BACKEND_NOT_SUPPORTED = "backend_not_supported"
    CUDA_BACKEND_NOT_ADMITTED = "cuda_backend_not_admitted"
    OUTPUT_DEVICE_MISMATCH = "output_device_mismatch"
    MEMORY_BUDGET_EXCEEDED = "memory_budget_exceeded"


@dataclass(frozen=True)
class ComputeRejection:
    code: ComputeRejectionCode
    message: str

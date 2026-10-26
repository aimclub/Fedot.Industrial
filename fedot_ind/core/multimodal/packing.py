"""Ordered, lossless 2-D transport for shaped multimodal tensors.

Packing preserves values and dtype. The explicit model boundary, unpacking,
converts to float32; neither direction shares storage with its input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from numbers import Integral

import numpy as np
import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.mapping import MODALITY_CAPABILITIES


@dataclass(frozen=True)
class ModalityLayout:
    """One named modality's shape, excluding the sample axis."""

    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Layout modality name must be a string.")
        if not self.name:
            raise ValueError("Layout modality name must be nonempty.")
        try:
            modality = MultimodalModality(self.name)
        except ValueError as exc:
            raise ValueError(f"Unsupported modality name {self.name!r}.") from exc
        if not isinstance(self.shape, (list, tuple)):
            raise TypeError(f"Shape for {self.name!r} must be a list or tuple.")
        if not self.shape or any(
            isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
            for size in self.shape
        ):
            raise ValueError(f"Shape for {self.name!r} must contain positive integers, not bools.")
        shape = tuple(int(size) for size in self.shape)
        spec = MODALITY_CAPABILITIES[modality].spec
        if len(shape) + 1 not in spec.allowed_ndim:
            raise ValueError(
                f"Shape for {self.name!r} has unsupported rank {len(shape)} "
                f"excluding the sample axis; full ranks are {spec.allowed_ndim}."
            )
        object.__setattr__(self, "shape", shape)

    @property
    def width(self) -> int:
        return prod(self.shape)


def parse_layout(payload: Sequence[Mapping | ModalityLayout]) -> tuple[ModalityLayout, ...]:
    """Validate ordered JSON records (or already parsed layout records)."""

    if not isinstance(payload, (list, tuple)):
        raise TypeError("modality_layout must be an ordered list or tuple of records.")
    if not payload:
        raise ValueError("modality_layout must contain at least one modality.")
    result = []
    names = set()
    for index, entry in enumerate(payload):
        if isinstance(entry, ModalityLayout):
            item = entry
        elif isinstance(entry, Mapping):
            if set(entry) != {"name", "shape"}:
                raise ValueError(f"Layout record {index} must have exactly 'name' and 'shape'.")
            item = ModalityLayout(name=entry["name"], shape=entry["shape"])
        else:
            raise TypeError(f"Layout record {index} must be a mapping or ModalityLayout.")
        if item.name in names:
            raise ValueError(f"Duplicate modality name {item.name!r} in layout.")
        names.add(item.name)
        result.append(item)
    return tuple(result)


def validate_packed_values(values: np.ndarray | torch.Tensor, width: int) -> None:
    """Require a nonempty finite real numeric matrix of exactly ``width`` columns."""

    if not isinstance(values, (np.ndarray, torch.Tensor)):
        raise TypeError("Packed values must be a numpy.ndarray or torch.Tensor.")
    if values.ndim != 2:
        raise ValueError(f"Packed values must be exactly 2-D, got shape {tuple(values.shape)}.")
    if values.shape[1] != width:
        raise ValueError(f"Packed values must have width {width}, got {values.shape[1]}.")
    if values.shape[0] == 0:
        raise ValueError("Packed values must contain at least one sample; empty batches are unsupported.")
    if isinstance(values, np.ndarray):
        if values.dtype.kind not in "iuf":
            raise TypeError("Packed values must have a real numeric dtype, not bool or object.")
        finite = np.isfinite(values).all()
    else:
        if values.layout != torch.strided or values.device.type == "meta":
            raise TypeError("Packed values must be a materialized strided tensor.")
        if values.dtype == torch.bool or values.is_complex() or values.is_quantized:
            raise TypeError("Packed values must have a real numeric dtype, not bool or complex.")
        finite = bool(torch.isfinite(values).all())
    if not finite:
        raise ValueError("Packed values must contain only finite values.")


def pack_modalities(
    modalities: Mapping[str | MultimodalModality, torch.Tensor],
    layout: Sequence[Mapping | ModalityLayout],
) -> torch.Tensor:
    """Flatten and concatenate in layout order, without reduction or coercion."""

    layout = parse_layout(layout)
    if not isinstance(modalities, Mapping):
        raise TypeError("modalities must be a mapping of names to torch.Tensor values.")
    named = {}
    for key, value in modalities.items():
        name = key.value if isinstance(key, MultimodalModality) else key
        if not isinstance(name, str):
            raise TypeError("Modality keys must be strings or MultimodalModality values.")
        if name in named:
            raise ValueError(f"Duplicate modality name {name!r} in modalities.")
        named[name] = value
    expected = {item.name for item in layout}
    if set(named) != expected:
        raise ValueError(
            f"Modality names must exactly match layout: missing={sorted(expected - set(named))}, "
            f"unknown={sorted(set(named) - expected)}."
        )
    flattened = []
    n_samples = dtype = device = None
    for item in layout:
        tensor = named[item.name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Modality {item.name!r} must be a torch.Tensor.")
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise TypeError(f"Modality {item.name!r} must be a materialized strided tensor.")
        if tuple(tensor.shape[1:]) != item.shape or tensor.ndim != len(item.shape) + 1:
            raise ValueError(
                f"Modality {item.name!r} must have shape (samples, {item.shape}), "
                f"got {tuple(tensor.shape)}."
            )
        if n_samples is None:
            n_samples, dtype, device = tensor.shape[0], tensor.dtype, tensor.device
        if tensor.shape[0] != n_samples:
            raise ValueError("All modalities must have the same number of samples.")
        if tensor.dtype != dtype or tensor.device != device:
            raise ValueError("Lossless packing requires the same dtype and device for all modalities.")
        flat = tensor.detach().reshape(n_samples, item.width)
        validate_packed_values(flat, item.width)
        flattened.append(flat)
    return torch.cat(flattened, dim=1).clone()


def unpack_modalities(
    values: np.ndarray | torch.Tensor,
    layout: Sequence[Mapping | ModalityLayout],
    target: np.ndarray | torch.Tensor | None = None,
) -> MultimodalDataBundle:
    """Restore named shaped modalities as independent, finite float32 tensors."""

    layout = parse_layout(layout)
    validate_packed_values(values, sum(item.width for item in layout))
    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(dtype=torch.float32).clone()
    else:
        tensor = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Packed values must remain finite after float32 conversion.")
    if target is not None:
        if isinstance(target, torch.Tensor):
            target = target.detach().to(device=tensor.device).clone()
        elif isinstance(target, np.ndarray):
            target = torch.from_numpy(target.copy()).to(device=tensor.device)
        else:
            raise TypeError("target must be a numpy.ndarray or torch.Tensor.")
    modalities = {}
    start = 0
    for item in layout:
        stop = start + item.width
        modalities[MultimodalModality(item.name)] = (
            tensor[:, start:stop].reshape(tensor.shape[0], *item.shape).clone()
        )
        start = stop
    return MultimodalDataBundle(modalities=modalities, target=target)

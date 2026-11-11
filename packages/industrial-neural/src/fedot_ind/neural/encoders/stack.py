"""Family encoder stacks keyed by arbitrary tensor stream names."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import torch
from torch import nn

from fedot_ind.neural.encoders.builder import build_encoder
from fedot_ind.neural.encoders.config import EncoderConfig

T = TypeVar("T")


def _stream_map(entries: Mapping[str, T] | Sequence[tuple[str, T]]) -> dict[str, T]:
    result: dict[str, T] = {}
    for name, value in entries.items() if isinstance(entries, Mapping) else entries:
        if not isinstance(name, str):
            raise TypeError("Stream names must be strings.")
        # Strip str subclasses without requiring domain-specific Enum.value.
        name = name[:]
        if not name or "." in name:
            raise ValueError("Stream names must be non-empty and must not contain '.'.")
        if name in result:
            raise ValueError(f"Duplicate stream definition: {name!r}.")
        result[name] = value
    return result


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class EncoderStack(nn.Module):
    """Apply one encoder per named stream, preserving registration order."""

    def __init__(
        self,
        encoder_configs: Mapping[str, EncoderConfig] | Sequence[tuple[str, EncoderConfig]],
    ):
        super().__init__()
        self.encoder_configs = _stream_map(encoder_configs)
        if not self.encoder_configs:
            raise ValueError("EncoderStack requires at least one encoder config.")
        d_models = {config.d_model for config in self.encoder_configs.values()}
        if len(d_models) != 1:
            raise ValueError(
                f"All stream encoders must share the same d_model. Got: {sorted(d_models)}."
            )
        self.embedding_dim = next(iter(d_models))
        self.stream_names = tuple(self.encoder_configs)
        self._stream_configs = self.encoder_configs
        self.encoders = nn.ModuleDict({
            name: build_encoder(config) for name, config in self.encoder_configs.items()
        })

    def forward(
        self,
        streams: Mapping[str, torch.Tensor],
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, Any]]:
        tensors = _stream_map(streams)
        missing = [name for name in self.stream_names if name not in tensors]
        if missing:
            raise ValueError(f"Encoder stack input does not contain required streams: {missing}.")

        batch_sizes = set()
        for name in self.stream_names:
            tensor = tensors[name]
            config = self._stream_configs[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Stream {name!r} must be torch.Tensor.")
            if tensor.ndim != config.input_rank:
                raise ValueError(f"Stream {name!r} expected rank {config.input_rank}, got {tensor.ndim}.")
            width = config.in_channels if config.in_channels is not None else config.in_features
            if tensor.shape[1] != width:
                raise ValueError(f"Stream {name!r} expected axis 1 size {width}, got {tensor.shape[1]}.")
            if any(size == 0 for size in tensor.shape[1:]):
                raise ValueError(f"Stream {name!r} must have non-empty feature axes.")
            batch_sizes.add(tensor.shape[0])
        if len(batch_sizes) != 1:
            raise ValueError("All streams must have the same batch size.")

        embeddings = {name: self.encoders[name](tensors[name]) for name in self.stream_names}
        if not return_aux:
            return embeddings
        return embeddings, {
            "active_streams": list(self.stream_names),
            "embedding_dim": self.embedding_dim,
            "num_parameters": {
                "total": _count_parameters(self),
                "per_stream": {
                    name: _count_parameters(self.encoders[name]) for name in self.stream_names
                },
            },
            "shapes": {
                "input": {name: tuple(tensors[name].shape) for name in self.stream_names},
                "output": {name: tuple(embeddings[name].shape) for name in self.stream_names},
            },
        }

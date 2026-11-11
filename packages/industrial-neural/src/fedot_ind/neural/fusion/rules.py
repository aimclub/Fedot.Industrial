"""Tensor-only fusion invariants, independent of modality registries."""
from collections.abc import Sequence
from typing import Any

import torch


def validate_positive_int(name: str, value: int, min_value: int = 1) -> None:
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}.")


def validate_choice(name: str, value: Any, allowed: Sequence[Any]) -> None:
    if value not in allowed:
        options = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"Unknown {name}={value!r}. Expected one of: {options}.")


def validate_divisible(dividend_name: str, dividend: int, divisor_name: str, divisor: int) -> None:
    if dividend % divisor != 0:
        raise ValueError(f"{dividend_name}={dividend} must be divisible by {divisor_name}={divisor}.")


def validate_embeddings_count(embeddings: tuple[torch.Tensor, ...], expected_count: int,
                              label: str = "embeddings") -> None:
    if len(embeddings) != expected_count:
        raise ValueError(f"Expected {expected_count} {label}, got {len(embeddings)}.")


def validate_stacked_embeddings_shape(stacked_embeddings: torch.Tensor,
                                      expected_n_inputs: int, expected_d_model: int) -> None:
    _, n_inputs, d_model = stacked_embeddings.shape
    if n_inputs != expected_n_inputs:
        raise ValueError(f"Expected n_inputs={expected_n_inputs}, got {n_inputs}.")
    if d_model != expected_d_model:
        raise ValueError(f"Expected d_model={expected_d_model}, got {d_model}.")

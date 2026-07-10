"""Validation rules shared by fusion strategies."""

import torch


def validate_positive_int(name: str, value: int, min_value: int = 1) -> None:
    """Validate that integer hyperparameter is not below threshold."""

    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}.")


def validate_embeddings_count(
    embeddings: tuple[torch.Tensor, ...],
    expected_count: int,
    label: str = "embeddings",
) -> None:
    """Validate number of provided embeddings."""

    if len(embeddings) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {label}, got {len(embeddings)}."
        )


def validate_stacked_embeddings_shape(
    stacked_embeddings: torch.Tensor,
    expected_n_inputs: int,
    expected_d_model: int,
) -> None:
    """Validate stacked embedding shape `(batch, n_inputs, d_model)`."""

    _, n_inputs, d_model = stacked_embeddings.shape

    if n_inputs != expected_n_inputs:
        raise ValueError(
            f"Expected n_inputs={expected_n_inputs}, got {n_inputs}."
        )
    if d_model != expected_d_model:
        raise ValueError(
            f"Expected d_model={expected_d_model}, got {d_model}."
        )

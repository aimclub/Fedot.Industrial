"""Aligned mini-batch helpers for :class:`MultimodalDataBundle`."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle


class MultimodalBundleIndexDataset(Dataset):
    """Dataset of sample indices into a fixed multimodal bundle."""

    def __init__(self, bundle: MultimodalDataBundle, *, require_target: bool = False):
        if require_target and bundle.target is None:
            raise ValueError("Training/validation bundles must include a target tensor.")
        self.bundle = bundle
        self._n_samples = bundle.n_samples

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, index: int) -> int:
        return int(index)


def select_bundle_indices(
    bundle: MultimodalDataBundle,
    indices: Sequence[int] | torch.Tensor,
) -> MultimodalDataBundle:
    """Return a new bundle containing the selected sample indices."""

    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    if index_tensor.ndim != 1:
        raise ValueError(f"indices must be 1-D, got shape {tuple(index_tensor.shape)}.")
    if index_tensor.numel() == 0:
        raise ValueError("indices must contain at least one sample.")

    return bundle.replace(
        modalities={
            modality: tensor.index_select(0, index_tensor.to(device=tensor.device))
            for modality, tensor in bundle.modalities.items()
        },
        target=(
            None
            if bundle.target is None
            else bundle.target.index_select(0, index_tensor.to(device=bundle.target.device))
        ),
    )


def split_bundle_by_fraction(
    bundle: MultimodalDataBundle,
    *,
    validation_fraction: float,
    seed: int | None = None,
) -> tuple[MultimodalDataBundle, MultimodalDataBundle]:
    """Split a bundle into train/validation subsets by sample indices."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in (0, 1), got {validation_fraction}."
        )
    if bundle.target is None:
        raise ValueError("Cannot split a bundle without a target tensor.")

    n_samples = bundle.n_samples
    n_val = max(1, int(round(n_samples * validation_fraction)))
    if n_val >= n_samples:
        raise ValueError(
            f"validation_fraction={validation_fraction} leaves no training samples "
            f"for n_samples={n_samples}."
        )

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator)
    val_indices = permutation[:n_val]
    train_indices = permutation[n_val:]
    return (
        select_bundle_indices(bundle, train_indices),
        select_bundle_indices(bundle, val_indices),
    )


def collate_bundle_indices(
    indices: Sequence[int],
    *,
    bundle: MultimodalDataBundle,
    device: torch.device | str | None = None,
) -> MultimodalDataBundle:
    """Collate sample indices into a mini-batch bundle on ``device``."""

    batch = select_bundle_indices(bundle, indices)
    if device is None:
        return batch
    return batch.to(device=device)


def make_bundle_dataloader(
    bundle: MultimodalDataBundle,
    *,
    batch_size: int,
    shuffle: bool = False,
    device: torch.device | str | None = None,
    seed: int | None = None,
    drop_last: bool = False,
    require_target: bool = False,
) -> DataLoader:
    """Build a DataLoader that yields aligned multimodal mini-batches."""

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    dataset = MultimodalBundleIndexDataset(bundle, require_target=require_target)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        collate_fn=lambda indices: collate_bundle_indices(
            indices,
            bundle=bundle,
            device=device,
        ),
    )


def iter_bundle_batches(
    bundle: MultimodalDataBundle,
    *,
    batch_size: int,
    shuffle: bool = False,
    device: torch.device | str | None = None,
    seed: int | None = None,
    drop_last: bool = False,
    require_target: bool = False,
) -> Iterator[MultimodalDataBundle]:
    """Iterate aligned multimodal mini-batches."""

    loader = make_bundle_dataloader(
        bundle,
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
        seed=seed,
        drop_last=drop_last,
        require_target=require_target,
    )
    yield from loader


__all__ = [
    "MultimodalBundleIndexDataset",
    "collate_bundle_indices",
    "iter_bundle_batches",
    "make_bundle_dataloader",
    "select_bundle_indices",
    "split_bundle_by_fraction",
]

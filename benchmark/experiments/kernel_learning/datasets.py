from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from benchmark.experiments.kernel_learning.controls import apply_optional_limit
from benchmark.industrial import DatasetSpec, discover_local_ucr_datasets


class KernelLearningDatasetValidationError(ValueError):
    """Expected validation failure for Kernel Learning dataset selection."""


class KernelLearningCustomDatasetPolicy(str, Enum):
    UCR_ONLY = "ucr_only"
    ALLOW_LOCAL = "allow_local"


@dataclass(frozen=True)
class ResolvedUCRDataset:
    name: str
    origin: str
    download_if_missing: bool


def normalize_custom_dataset_policy(value: KernelLearningCustomDatasetPolicy |
                                    str) -> KernelLearningCustomDatasetPolicy:
    if isinstance(value, KernelLearningCustomDatasetPolicy):
        return value
    try:
        return KernelLearningCustomDatasetPolicy(str(value).strip().lower())
    except ValueError as exc:
        supported = ", ".join(item.value for item in KernelLearningCustomDatasetPolicy)
        raise KernelLearningDatasetValidationError(
            f"Unsupported custom_dataset_policy={value!r}. Supported values: {supported}."
        ) from exc


def resolve_ucr_dataset_plans(
        *,
        data_root: str | Path,
        datasets: Sequence[str] = (),
        dataset_limit: int | None = None,
        allowed_dataset_names: Sequence[str] | None = None,
        custom_dataset_policy: KernelLearningCustomDatasetPolicy | str = KernelLearningCustomDatasetPolicy.UCR_ONLY,
) -> tuple[ResolvedUCRDataset, ...]:
    policy = normalize_custom_dataset_policy(custom_dataset_policy)
    allowed_names = None if allowed_dataset_names is None else tuple(str(item) for item in allowed_dataset_names)
    if datasets:
        requested = tuple(str(item) for item in datasets)
        plans = tuple(
            _resolve_requested_ucr_dataset(
                name,
                data_root=data_root,
                allowed_dataset_names=allowed_names,
                custom_dataset_policy=policy,
            )
            for name in requested
        )
        return tuple(apply_optional_limit(plans, dataset_limit))

    discovered = discover_local_ucr_datasets(data_root, allowed_names=allowed_names)
    return tuple(
        ResolvedUCRDataset(name=str(name), origin="ucr", download_if_missing=True)
        for name in apply_optional_limit(discovered, dataset_limit)
    )


def build_ucr_dataset_spec(plan: ResolvedUCRDataset, *, data_root: str | Path) -> DatasetSpec:
    return DatasetSpec(
        benchmark="ucr",
        dataset_name=plan.name,
        adapter_options={
            "local_data_root": str(data_root),
            "download_if_missing": bool(plan.download_if_missing),
            "dataset_origin": plan.origin,
        },
    )


def _resolve_requested_ucr_dataset(
        name: str,
        *,
        data_root: str | Path,
        allowed_dataset_names: Sequence[str] | None,
        custom_dataset_policy: KernelLearningCustomDatasetPolicy,
) -> ResolvedUCRDataset:
    if allowed_dataset_names is None or name in set(allowed_dataset_names):
        return ResolvedUCRDataset(name=name, origin="ucr", download_if_missing=True)

    local_custom_names = set(discover_local_ucr_datasets(data_root, allowed_names=None))
    if name in local_custom_names:
        if custom_dataset_policy is KernelLearningCustomDatasetPolicy.ALLOW_LOCAL:
            return ResolvedUCRDataset(name=name, origin="local_custom", download_if_missing=False)
        raise KernelLearningDatasetValidationError(
            f"Dataset {name!r} is not in the configured UCR allow-list. "
            "Set custom_dataset_policy='allow_local' to run a local custom split."
        )

    preview = ", ".join(sorted(allowed_dataset_names)[:5])
    suffix = "..." if len(allowed_dataset_names) > 5 else ""
    raise KernelLearningDatasetValidationError(
        f"Dataset {name!r} is neither a known UCR dataset nor a local custom split under {Path(data_root)}. "
        f"Known UCR examples: {preview}{suffix}."
    )


__all__ = [
    "KernelLearningCustomDatasetPolicy",
    "KernelLearningDatasetValidationError",
    "ResolvedUCRDataset",
    "build_ucr_dataset_spec",
    "normalize_custom_dataset_policy",
    "resolve_ucr_dataset_plans",
]

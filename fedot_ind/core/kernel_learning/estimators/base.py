from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from fedot_ind.core.kernel_learning.cache import InMemoryKernelCache, KernelCachePolicy
from fedot_ind.core.kernel_learning.contracts import (
    FeatureInput,
    KernelBundle,
    KernelConfigValidationError,
    KernelMatrixPolicy,
    KernelTaskType,
    TargetInput,
)
from fedot_ind.core.kernel_learning.generators import DEFAULT_GENERATOR_NAMES, create_feature_generator
from fedot_ind.core.kernel_learning.kernels import KernelMatrixBuilder
from fedot_ind.core.kernel_learning.selection import (
    KernelImportanceConfig,
    MKLObjectiveConfig,
    SparseMKLSelector,
    combine_kernels,
    select_significant_generators,
)


BASE_KERNEL_PARAMETER_NAMES = (
    "generator_names",
    "kernel",
    "gamma",
    "normalize",
    "center",
    "psd_correction",
    "psd_tol",
    "kernel_approximation",
    "nystrom_components",
    "complexity_penalty",
    "redundancy_penalty",
    "min_weight",
    "target_gamma",
    "selector_optimizer",
    "selector_max_iter",
    "selector_tol",
    "selector_step_size",
    "importance_threshold",
    "importance_fallback_top_n",
    "importance_max_union_size",
    "torch_device",
    "kernel_cache_enabled",
    "kernel_cache_namespace",
)


@dataclass(frozen=True)
class KernelEnsembleRuntimeConfig:
    generator_names: tuple[str, ...]
    kernel_policy: KernelMatrixPolicy
    selector_config: MKLObjectiveConfig
    importance_config: KernelImportanceConfig
    torch_device: Any
    cache_policy: KernelCachePolicy


def collect_kernel_base_params(values: Mapping[str, Any]) -> dict[str, Any]:
    return {name: values[name] for name in BASE_KERNEL_PARAMETER_NAMES}


class KernelEnsembleBase(BaseEstimator):
    task_type = "supervised"

    def __init__(
            self,
            generator_names: tuple[str, ...] | list[str] | None = None,
            kernel: str = "rbf",
            gamma: str | float = "scale",
            normalize: str | None = "trace",
            center: bool = False,
            psd_correction: str | None = "clip",
            psd_tol: float = 1e-8,
            kernel_approximation: str | None = None,
            nystrom_components: int | None = None,
            complexity_penalty: float = 0.01,
            redundancy_penalty: float = 0.05,
            min_weight: float = 0.05,
            target_gamma: str | float = "scale",
            selector_optimizer: str = "projected_gradient",
            selector_max_iter: int = 100,
            selector_tol: float = 1e-6,
            selector_step_size: float = 0.2,
            importance_threshold: float = 0.05,
            importance_fallback_top_n: int = 1,
            importance_max_union_size: int = 3,
            torch_device: Any = "auto",
            kernel_cache_enabled: bool = True,
            kernel_cache_namespace: str = "kernel_ensemble",
    ):
        self.generator_names = generator_names
        self.kernel = kernel
        self.gamma = gamma
        self.normalize = normalize
        self.center = center
        self.psd_correction = psd_correction
        self.psd_tol = psd_tol
        self.kernel_approximation = kernel_approximation
        self.nystrom_components = nystrom_components
        self.complexity_penalty = complexity_penalty
        self.redundancy_penalty = redundancy_penalty
        self.min_weight = min_weight
        self.target_gamma = target_gamma
        self.selector_optimizer = selector_optimizer
        self.selector_max_iter = selector_max_iter
        self.selector_tol = selector_tol
        self.selector_step_size = selector_step_size
        self.importance_threshold = importance_threshold
        self.importance_fallback_top_n = importance_fallback_top_n
        self.importance_max_union_size = importance_max_union_size
        self.torch_device = torch_device
        self.kernel_cache_enabled = kernel_cache_enabled
        self.kernel_cache_namespace = kernel_cache_namespace

    def _resolve_generator_names(self) -> tuple[str, ...]:
        if self.generator_names is None:
            names = DEFAULT_GENERATOR_NAMES
        else:
            names = tuple(self.generator_names)
        if not names:
            raise KernelConfigValidationError("Kernel ensemble requires at least one feature generator.")
        return names

    def _build_runtime_config(self) -> KernelEnsembleRuntimeConfig:
        return KernelEnsembleRuntimeConfig(
            generator_names=self._resolve_generator_names(),
            kernel_policy=KernelMatrixPolicy(
                kernel=self.kernel,
                gamma=self.gamma,
                normalize=self.normalize,
                center=self.center,
                psd_correction=self.psd_correction,
                psd_tol=self.psd_tol,
                approximation=self.kernel_approximation,
                nystrom_components=self.nystrom_components,
            ).normalized_policy,
            selector_config=MKLObjectiveConfig(
                optimizer=self.selector_optimizer,
                max_iter=self.selector_max_iter,
                tol=self.selector_tol,
                step_size=self.selector_step_size,
                complexity_penalty=self.complexity_penalty,
                redundancy_penalty=self.redundancy_penalty,
                min_weight=self.min_weight,
            ),
            importance_config=KernelImportanceConfig(
                weight_threshold=self.importance_threshold,
                fallback_top_n=self.importance_fallback_top_n,
                max_union_size=self.importance_max_union_size,
            ),
            torch_device=self.torch_device,
            cache_policy=KernelCachePolicy(
                enabled=bool(self.kernel_cache_enabled),
                namespace=str(self.kernel_cache_namespace),
            ),
        )

    def _kernel_task_value(self) -> str:
        if isinstance(self.task_type, KernelTaskType):
            return self.task_type.value
        return str(self.task_type)

    def _fit_kernel_layer(self, X: FeatureInput, y: TargetInput) -> np.ndarray:
        runtime_config = self._build_runtime_config()
        self.runtime_config_ = runtime_config
        self.generator_names_ = runtime_config.generator_names
        self.generators_ = []
        self.kernel_builders_ = {}
        self.kernel_bundles_: list[KernelBundle] = []
        self.kernel_cache_ = InMemoryKernelCache() if runtime_config.cache_policy.enabled else None

        for name in self.generator_names_:
            generator = create_feature_generator(name, torch_device=runtime_config.torch_device)
            feature_bundle = generator.fit_transform(X, y, task_type=self._kernel_task_value())
            kernel_policy = runtime_config.kernel_policy
            builder = KernelMatrixBuilder(
                kernel=kernel_policy.kernel,
                gamma=kernel_policy.gamma,
                normalize=kernel_policy.normalize,
                center=kernel_policy.center,
                psd_correction=kernel_policy.psd_correction,
                psd_tol=kernel_policy.psd_tol,
                approximation=kernel_policy.approximation,
                nystrom_components=kernel_policy.nystrom_components,
                cache_policy=runtime_config.cache_policy,
                cache=self.kernel_cache_,
            )
            kernel_bundle = builder.fit_transform(
                feature_bundle.features,
                name=feature_bundle.name,
                train_features=feature_bundle.features,
            )
            kernel_bundle = replace(
                kernel_bundle,
                diagnostics={
                    **kernel_bundle.diagnostics,
                    "feature_generator": dict(feature_bundle.diagnostics),
                },
            )
            self.generators_.append(generator)
            self.kernel_builders_[feature_bundle.name] = builder
            self.kernel_bundles_.append(kernel_bundle)

        selector_config = runtime_config.selector_config
        selector = SparseMKLSelector(
            complexity_penalty=selector_config.complexity_penalty,
            redundancy_penalty=selector_config.redundancy_penalty,
            min_weight=selector_config.min_weight,
            target_gamma=self.target_gamma,
            optimizer=selector_config.optimizer,
            max_iter=selector_config.max_iter,
            tol=selector_config.tol,
            step_size=selector_config.step_size,
        )
        self.selection_report_ = selector.fit(self.kernel_bundles_, y, task_type=self._kernel_task_value())
        self.selected_generators_ = self.selection_report_.selected_generators
        self.selected_weights_ = self.selection_report_.selected_weights
        self.kernel_importance_ = select_significant_generators(
            self.selection_report_,
            runtime_config.importance_config,
        )
        self.important_generators_ = self.kernel_importance_.selected_generators
        self.important_weights_ = self.kernel_importance_.selected_weights
        self.fit_diagnostics_ = {
            "task_type": self._kernel_task_value(),
            "generator_names": self.generator_names_,
            "selected_generators": self.selected_generators_,
            "important_generators": self.important_generators_,
            "selector": dict(self.selection_report_.diagnostics),
            "weight_source": "selection_report",
            "kernel_cache": {
                "enabled": bool(runtime_config.cache_policy.enabled),
                "namespace": runtime_config.cache_policy.namespace,
                "size": 0 if self.kernel_cache_ is None else self.kernel_cache_.size,
            },
            "kernel_bundles": tuple(bundle.to_dict() for bundle in self.kernel_bundles_),
        }
        return self._combine_train_kernels()

    def _combine_train_kernels(self) -> np.ndarray:
        selected = self._selected_bundle_weight_pairs()
        return combine_kernels([bundle.train_kernel for bundle, _ in selected], [weight for _, weight in selected])

    def _combine_test_kernels(self, X: Any) -> np.ndarray:
        test_kernels = []
        weights = []
        self.last_test_kernel_bundles_ = []
        for bundle, weight in self._selected_bundle_weight_pairs():
            generator = self._generator_by_name(bundle.name)
            feature_bundle = generator.transform(X)
            builder = self.kernel_builders_[bundle.name]
            test_bundle = builder.build_test_bundle(feature_bundle.features, bundle)
            self.last_test_kernel_bundles_.append(test_bundle)
            test_kernels.append(test_bundle.test_kernel)
            weights.append(weight)
        return combine_kernels(test_kernels, weights)

    def _selected_bundle_weight_pairs(self) -> list[tuple[KernelBundle, float]]:
        weights_by_name = dict(zip(self.selection_report_.generator_names, self.selection_report_.weights))
        return [
            (bundle, float(weights_by_name[bundle.name]))
            for bundle in self.kernel_bundles_
            if float(weights_by_name[bundle.name]) > 0.0
        ]

    def _generator_by_name(self, name: str):
        for generator in self.generators_:
            if generator.name == name:
                return generator
        raise ValueError(f"Fitted generator not found: {name}")

    def _clone_head(self, head):
        return deepcopy(head)

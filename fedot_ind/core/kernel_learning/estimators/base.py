from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from fedot_ind.core.kernel_learning.contracts import KernelBundle
from fedot_ind.core.kernel_learning.generators import DEFAULT_GENERATOR_NAMES, create_feature_generator
from fedot_ind.core.kernel_learning.kernels import KernelMatrixBuilder
from fedot_ind.core.kernel_learning.selection import (
    KernelImportanceConfig,
    SparseMKLSelector,
    combine_kernels,
    select_significant_generators,
)


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
            complexity_penalty: float = 0.01,
            redundancy_penalty: float = 0.05,
            min_weight: float = 0.05,
            target_gamma: str | float = "scale",
            importance_threshold: float = 0.15,
            importance_fallback_top_n: int = 1,
            importance_max_union_size: int = 3,
    ):
        self.generator_names = generator_names
        self.kernel = kernel
        self.gamma = gamma
        self.normalize = normalize
        self.center = center
        self.psd_correction = psd_correction
        self.psd_tol = psd_tol
        self.complexity_penalty = complexity_penalty
        self.redundancy_penalty = redundancy_penalty
        self.min_weight = min_weight
        self.target_gamma = target_gamma
        self.importance_threshold = importance_threshold
        self.importance_fallback_top_n = importance_fallback_top_n
        self.importance_max_union_size = importance_max_union_size

    def _resolve_generator_names(self) -> tuple[str, ...]:
        if self.generator_names is None:
            return DEFAULT_GENERATOR_NAMES
        return tuple(self.generator_names)

    def _fit_kernel_layer(self, X: Any, y: Any) -> np.ndarray:
        self.generator_names_ = self._resolve_generator_names()
        self.generators_ = []
        self.kernel_builders_ = {}
        self.kernel_bundles_: list[KernelBundle] = []

        for name in self.generator_names_:
            generator = create_feature_generator(name)
            feature_bundle = generator.fit_transform(X, y, task_type=self.task_type)
            builder = KernelMatrixBuilder(
                kernel=self.kernel,
                gamma=self.gamma,
                normalize=self.normalize,
                center=self.center,
                psd_correction=self.psd_correction,
                psd_tol=self.psd_tol,
            )
            kernel_bundle = builder.fit_transform(
                feature_bundle.features,
                name=feature_bundle.name,
                train_features=feature_bundle.features,
            )
            self.generators_.append(generator)
            self.kernel_builders_[feature_bundle.name] = builder
            self.kernel_bundles_.append(kernel_bundle)

        selector = SparseMKLSelector(
            complexity_penalty=self.complexity_penalty,
            redundancy_penalty=self.redundancy_penalty,
            min_weight=self.min_weight,
            target_gamma=self.target_gamma,
        )
        self.selection_report_ = selector.fit(self.kernel_bundles_, y, task_type=self.task_type)
        self.selected_generators_ = self.selection_report_.selected_generators
        self.selected_weights_ = self.selection_report_.selected_weights
        self.kernel_importance_ = select_significant_generators(
            self.selection_report_,
            KernelImportanceConfig(
                weight_threshold=self.importance_threshold,
                fallback_top_n=self.importance_fallback_top_n,
                max_union_size=self.importance_max_union_size,
            ),
        )
        self.important_generators_ = self.kernel_importance_.selected_generators
        self.important_weights_ = self.kernel_importance_.selected_weights
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

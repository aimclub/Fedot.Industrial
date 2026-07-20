from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from fedot_ind.core.kernel_learning.contracts import FeatureBundle, FeatureInput, TargetInput
from fedot_ind.core.kernel_learning.generators.base import (
    KernelFeatureGeneratorMixin,
    OperationSpec,
    call_transform,
    features_to_fedot_input_data,
    load_operation_class,
    normalize_feature_matrix,
    normalize_time_series_tensor,
    operation_params,
    resolve_torch_device,
    to_fedot_input_data,
    unwrap_operation_output,
)
from fedot_ind.core.kernel_learning.generators.specs import _DEFAULT_STATISTICAL_PARAMS, torch_quantile_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratorBudgetPolicy:
    max_samples: int | None = None
    max_timestamps: int | None = None
    max_cells: int | None = None
    fallback_generator: str = "identity"

    def allows(self, X: Any) -> tuple[bool, dict[str, Any]]:
        tensor = normalize_time_series_tensor(X)
        n_samples = int(tensor.shape[0])
        n_channels = int(tensor.shape[1])
        n_timestamps = int(tensor.shape[2])
        n_cells = int(np.prod(tensor.shape))
        diagnostics = {
            "n_samples": n_samples,
            "n_channels": n_channels,
            "n_timestamps": n_timestamps,
            "n_cells": n_cells,
            "max_samples": self.max_samples,
            "max_timestamps": self.max_timestamps,
            "max_cells": self.max_cells,
        }
        blocked_reasons = []
        if self.max_samples is not None and n_samples > self.max_samples:
            blocked_reasons.append("max_samples")
        if self.max_timestamps is not None and n_timestamps > self.max_timestamps:
            blocked_reasons.append("max_timestamps")
        if self.max_cells is not None and n_cells > self.max_cells:
            blocked_reasons.append("max_cells")
        diagnostics["blocked_reasons"] = tuple(blocked_reasons)
        return not blocked_reasons, diagnostics


@dataclass
class RepositoryFeatureGeneratorAdapter(KernelFeatureGeneratorMixin):
    name: str
    operation_specs: Sequence[OperationSpec]
    torch_device: Any = "auto"
    task_type_: str | None = None
    operations_: list[Any] = field(default_factory=list)
    train_features_: np.ndarray | None = None
    resolved_torch_device_: str | None = None

    def fit(self, X: FeatureInput, y: TargetInput | None = None, *, task_type: str = "classification"):
        self.task_type_ = task_type
        self.operations_ = [self._build_operation(
            spec) for spec in self.operation_specs]
        self.train_features_ = self._run_operations(X, y)
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        if not self.operations_:
            raise ValueError(
                f"Feature generator {self.name!r} must be fitted before transform.")
        features = self._run_operations(X, None)
        return self._bundle(features)

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: str = "classification",
    ) -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self._bundle(self.train_features_)

    def _build_operation(self, spec: OperationSpec):
        operation_cls = load_operation_class(spec)
        return operation_cls(operation_params(dict(spec.params)))

    def _run_operations(self, X: Any, y: Any | None) -> np.ndarray:
        if not self.operation_specs:
            raise ValueError(
                f"Feature generator {self.name!r} has no operation specs.")

        torch_device = self._resolve_torch_device_for_run()
        current = to_fedot_input_data(
            X,
            y,
            task_type=self.task_type_ or "classification",
            use_torch=self.operation_specs[0].use_torch,
            torch_device=torch_device,
        )
        raw_output: Any = None
        for index, (spec, operation) in enumerate(zip(self.operation_specs, self.operations_)):
            output = call_transform(operation, current)
            raw_output = unwrap_operation_output(output)
            next_index = index + 1
            if next_index < len(self.operation_specs):
                current = features_to_fedot_input_data(
                    raw_output,
                    current,
                    use_torch=self.operation_specs[next_index].use_torch,
                    torch_device=torch_device,
                )
        return normalize_feature_matrix(raw_output)

    def _resolve_torch_device_for_run(self):
        if any(spec.use_torch for spec in self.operation_specs):
            resolved = resolve_torch_device(self.torch_device)
            self.resolved_torch_device_ = str(resolved)
            return resolved
        self.resolved_torch_device_ = None
        return self.torch_device

    def _bundle(self, features: Any) -> FeatureBundle:
        matrix = normalize_feature_matrix(features)
        diagnostics = {
            "source": "fedot_industrial_operation",
            "operations": tuple(spec.name for spec in self.operation_specs),
            "n_features": int(matrix.shape[1]),
        }
        operation_params = {
            spec.name: dict(spec.params)
            for spec in self.operation_specs
            if spec.params
        }
        if operation_params:
            diagnostics["operation_params"] = operation_params
        for operation in self.operations_:
            logging_params = getattr(operation, "logging_params", None)
            if isinstance(logging_params, dict):
                diagnostics.update(logging_params)
        if len(self.operation_specs) == 1:
            diagnostics.update(dict(self.operation_specs[0].params))
        if self.resolved_torch_device_ is not None:
            diagnostics["torch_device"] = self.resolved_torch_device_
        return FeatureBundle(
            name=self.name,
            features=matrix,
            diagnostics=diagnostics,
        )


@dataclass
class BudgetedRepositoryFeatureGeneratorAdapter(RepositoryFeatureGeneratorAdapter):
    budget_policy: GeneratorBudgetPolicy = field(
        default_factory=GeneratorBudgetPolicy)
    fallback_generator_: Any | None = None
    budget_diagnostics_: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        allowed, diagnostics = self.budget_policy.allows(X)
        self.budget_diagnostics_ = diagnostics
        if not allowed:
            logger.info(
                "Falling back from generator %s because budget policy blocked execution: %s",
                self.name,
                diagnostics.get("blocked_reasons"),
            )
            return self._fit_fallback(X, y, task_type=task_type, reason="budget_exceeded")
        try:
            return super().fit(X, y, task_type=task_type)
        except Exception as ex:
            logger.exception(
                "Falling back from generator %s after operation failure.", self.name)
            return self._fit_fallback(
                X,
                y,
                task_type=task_type,
                reason=f"operation_unavailable:{ex.__class__.__name__}")

    def transform(self, X: Any) -> FeatureBundle:
        if self.fallback_generator_ is not None:
            bundle = self.fallback_generator_.transform(X)
            return FeatureBundle(
                name=self.name,
                features=bundle.features,
                diagnostics={
                    **bundle.diagnostics,
                    "source": "budgeted_fallback",
                    "requested_generator": self.name,
                    "budget": self.budget_diagnostics_,
                },
            )
        bundle = super().transform(X)
        return FeatureBundle(
            name=bundle.name,
            features=bundle.features,
            diagnostics={**bundle.diagnostics,
                         "budget": self.budget_diagnostics_},
        )

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)

    def _fit_fallback(self, X: Any, y: Any | None, *, task_type: str, reason: str):
        self.fallback_generator_ = build_lightweight_fallback(
            self.budget_policy.fallback_generator)
        self.fallback_generator_.fit(X, y, task_type=task_type)
        self.budget_diagnostics_ = {
            **self.budget_diagnostics_,
            "skipped": True,
            "skip_reason": reason,
            "fallback_generator": self.budget_policy.fallback_generator,
        }
        return self


@dataclass
class PipelineFeatureGeneratorAdapter(KernelFeatureGeneratorMixin):
    name: str
    pipeline_factory: Any
    task_type_: str | None = None
    pipeline_: Any | None = None
    train_features_: np.ndarray | None = None

    def fit(self, X: FeatureInput, y: TargetInput | None = None, *, task_type: str = "classification"):
        self.task_type_ = task_type
        self.pipeline_ = deepcopy(self.pipeline_factory).build()
        input_data = to_fedot_input_data(X, y, task_type=task_type)
        self.pipeline_.fit(input_data)
        prediction = self.pipeline_.predict(input_data)
        self.train_features_ = normalize_feature_matrix(
            unwrap_operation_output(prediction))
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        if self.pipeline_ is None:
            raise ValueError(
                f"Feature generator {self.name!r} must be fitted before transform.")
        input_data = to_fedot_input_data(
            X, task_type=self.task_type_ or "classification")
        prediction = self.pipeline_.predict(input_data)
        features = normalize_feature_matrix(
            unwrap_operation_output(prediction))
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "fedot_pipeline",
                         "n_features": int(features.shape[1])},
        )

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: str = "classification",
    ) -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        features = normalize_feature_matrix(self.train_features_)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "fedot_pipeline",
                         "n_features": int(features.shape[1])},
        )


class SummaryFeatureGenerator(RepositoryFeatureGeneratorAdapter):
    def __init__(
            self,
            name: str = "statistical_summary",
            params: dict[str, Any] | None = None,
            torch_device: Any = "auto",
    ):
        super().__init__(
            name=name,
            operation_specs=(
                torch_quantile_spec(params or _DEFAULT_STATISTICAL_PARAMS),
            ),
            torch_device=torch_device,
        )


def build_lightweight_fallback(name: str):
    from fedot_ind.core.kernel_learning.generators.lightweight import (
        IdentityFeatureGenerator,
        RandomProjectionEmbeddingFeatureGenerator,
        ShapeletFeatureGenerator,
    )

    if name == "identity":
        return IdentityFeatureGenerator()
    if name in {"statistical_summary", "statistical", "quantile_extractor", "quantile_extractor_torch"}:
        return SummaryFeatureGenerator(name=name)
    if name in {"embedding_extractor", "foundation_embedding"}:
        return RandomProjectionEmbeddingFeatureGenerator(name=name)
    if name in {"shapelet_extractor", "local_pattern_extractor"}:
        return ShapeletFeatureGenerator(name=name)
    return IdentityFeatureGenerator(name=f"{name}_fallback")


_build_lightweight_fallback = build_lightweight_fallback


__all__ = [
    "BudgetedRepositoryFeatureGeneratorAdapter",
    "GeneratorBudgetPolicy",
    "PipelineFeatureGeneratorAdapter",
    "RepositoryFeatureGeneratorAdapter",
    "SummaryFeatureGenerator",
    "build_lightweight_fallback",
]

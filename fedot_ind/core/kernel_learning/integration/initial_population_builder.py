from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from fedot_ind.core.kernel_learning.generators import OperationSpec, resolve_generator_operation_specs
from fedot_ind.core.kernel_learning.selection import KernelImportanceReport

CLASSIFICATION_HEADS = ("rf", "logit", "xgboost", "catboost", "dt", "mlp", "lgbm")
REGRESSION_HEADS = ("treg", "ridge", "xgbreg", "dtreg", "lgbmreg", "catboostreg", "lasso")
FORECASTING_HEADS = (
    "ridge",
    "lagged_ridge_forecaster",
    "okhs_fdmd_forecaster",
    "topo_forecaster",
    "lagged_forecaster",
)
SAFE_PREPROCESSORS = ("scaling", "normalization", "simple_imputation", "kernel_pca")


@dataclass(frozen=True)
class KernelInitialPipelineSpec:
    name: str
    kind: str
    generator_names: tuple[str, ...]
    operation_chains: tuple[tuple[OperationSpec, ...], ...]
    head_model: str
    source_weights: tuple[float, ...]

    @property
    def operation_names(self) -> tuple[str, ...]:
        names = []
        for chain in self.operation_chains:
            names.extend(spec.name for spec in chain)
        names.append(self.head_model)
        return tuple(names)


@dataclass
class KernelInitialPopulationBuilder:
    task_type: str
    head_model: str | None = None
    include_feature_union: bool = True
    max_union_size: int = 3
    materialize_basis: bool = True
    diagnostics_: dict = field(default_factory=dict, init=False)
    last_specs_: tuple[KernelInitialPipelineSpec, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self):
        if self.task_type not in ("classification", "regression", "forecasting", "ts_forecasting"):
            raise ValueError(f"Unsupported kernel warm-start task_type: {self.task_type}")
        if self.task_type == "ts_forecasting":
            self.task_type = "forecasting"
        if self.max_union_size < 1:
            raise ValueError("max_union_size must be at least 1.")

    @property
    def resolved_head_model(self) -> str:
        if self.head_model is not None:
            return self.head_model
        if self.task_type == "classification":
            return "rf"
        if self.task_type == "forecasting":
            return "ridge"
        return "treg"

    @property
    def task_head_candidates(self) -> tuple[str, ...]:
        return _task_heads(self.task_type)

    def build_specs(self, importance: KernelImportanceReport) -> tuple[KernelInitialPipelineSpec, ...]:
        specs = []
        skipped = {}
        valid_items = []

        for item in importance.items:
            operation_chain = resolve_generator_operation_specs(
                item.name,
                materialize_basis=self.materialize_basis,
            )
            if not operation_chain:
                skipped[item.name] = "no_fedot_operation_chain"
                continue
            valid_items.append((item, operation_chain))
            specs.append(
                KernelInitialPipelineSpec(
                    name=f"{item.name}_{self.resolved_head_model}",
                    kind="single",
                    generator_names=(item.name,),
                    operation_chains=(operation_chain,),
                    head_model=self.resolved_head_model,
                    source_weights=(float(item.weight),),
                )
            )

        union_items = valid_items[:self.max_union_size]
        if self.include_feature_union and len(union_items) > 1:
            generator_names = tuple(item.name for item, _ in union_items)
            specs.append(
                KernelInitialPipelineSpec(
                    name=f"{'_'.join(generator_names)}_{self.resolved_head_model}_union",
                    kind="union",
                    generator_names=generator_names,
                    operation_chains=tuple(chain for _, chain in union_items),
                    head_model=self.resolved_head_model,
                    source_weights=tuple(float(item.weight) for item, _ in union_items),
                )
            )

        self.last_specs_ = tuple(specs)
        self.diagnostics_ = {
            "skipped_generators": skipped,
            "include_feature_union": bool(self.include_feature_union),
            "max_union_size": int(self.max_union_size),
            "head_model": self.resolved_head_model,
        }
        return self.last_specs_

    def build_pipelines(self, importance: KernelImportanceReport):
        specs = self.build_specs(importance)
        return [self.build_pipeline_from_spec(spec) for spec in specs]

    def build_pipeline_builders(self, importance: KernelImportanceReport):
        specs = self.build_specs(importance)
        return [self.build_pipeline_builder_from_spec(spec) for spec in specs]

    def build_pipeline_builder_from_spec(self, spec: KernelInitialPipelineSpec):
        from fedot.core.pipelines.pipeline_builder import PipelineBuilder

        builder = PipelineBuilder()
        if spec.kind == "single":
            for operation in spec.operation_chains[0]:
                builder.add_node(operation.name, params=dict(operation.params))
            builder.add_node(spec.head_model)
            return builder

        if spec.kind == "union":
            for branch_idx, chain in enumerate(spec.operation_chains):
                for operation in chain:
                    builder.add_node(operation.name, branch_idx=branch_idx, params=dict(operation.params))
            return builder.join_branches(spec.head_model)

        raise ValueError(f"Unsupported kernel initial pipeline kind: {spec.kind}")

    def build_pipeline_from_spec(self, spec: KernelInitialPipelineSpec):
        return self.build_pipeline_builder_from_spec(spec).build()

    def restrict_available_operations(
            self,
            available_operations: Sequence[str] | None,
            specs: Sequence[KernelInitialPipelineSpec] | None = None,
    ) -> list[str] | None:
        return narrow_kernel_learning_search_space(
            available_operations=available_operations,
            specs=specs if specs is not None else self.last_specs_,
            task_type=self.task_type,
            head_model=self.resolved_head_model,
            diagnostics=self.diagnostics_,
        )


def narrow_kernel_learning_search_space(
        available_operations: Sequence[str] | None,
        specs: Sequence[KernelInitialPipelineSpec],
        *,
        task_type: str,
        head_model: str,
        diagnostics: dict | None = None,
) -> list[str] | None:
    if available_operations is None:
        return None
    if not specs:
        if diagnostics is not None:
            diagnostics["search_space_narrowing"] = "skipped_empty_specs"
        return list(available_operations)

    source_operations = _deduplicate(available_operations)
    source_set = set(source_operations)
    task_heads = _task_heads(task_type)
    valid_heads = [head_model] if head_model in task_heads else []
    valid_heads.extend(
        operation
        for operation in source_operations
        if operation in task_heads and operation != head_model
    )

    if not valid_heads:
        if diagnostics is not None:
            diagnostics["search_space_narrowing"] = "skipped_no_valid_head"
        return source_operations

    generator_operations = _deduplicate(
        operation
        for spec in specs
        for operation in spec.operation_names
        if operation != spec.head_model and operation != "topological_extractor"
    )
    kept_preprocessors = [
        operation
        for operation in SAFE_PREPROCESSORS
        if operation in source_set
    ]
    narrowed = _deduplicate([*generator_operations, *valid_heads, *kept_preprocessors])
    if diagnostics is not None:
        diagnostics["search_space_narrowing"] = "applied"
        diagnostics["available_operations_before"] = len(source_operations)
        diagnostics["available_operations_after"] = len(narrowed)
    return narrowed


def _deduplicate(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _task_heads(task_type: str) -> tuple[str, ...]:
    normalized = "forecasting" if task_type == "ts_forecasting" else task_type
    if normalized == "classification":
        return CLASSIFICATION_HEADS
    if normalized == "forecasting":
        return FORECASTING_HEADS
    return REGRESSION_HEADS

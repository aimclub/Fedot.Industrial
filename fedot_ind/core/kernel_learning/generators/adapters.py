from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Sequence

import numpy as np

from fedot_ind.core.kernel_learning.contracts import FeatureBundle, FeatureInput, KernelBundle, KernelTaskType, TargetInput

DEFAULT_GENERATOR_NAMES = (
    "quantile_extractor",
    "wavelet_extractor",
    "fourier_extractor",
    "eigen_extractor",
)
BASIS_ONLY_GENERATORS = frozenset(("wavelet_basis", "fourier_basis", "eigen_basis"))


@dataclass(frozen=True)
class OperationSpec:
    name: str
    module_path: str
    class_name: str
    params: dict[str, Any] = field(default_factory=dict)
    use_torch: bool = False


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


def normalize_feature_matrix(values: Any) -> np.ndarray:
    array = _to_numpy(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_time_series_tensor(values: Any) -> np.ndarray:
    array = _to_numpy(values).astype(float, copy=False)
    if array.ndim == 0:
        raise ValueError("X must contain at least one sample.")
    if array.ndim == 1:
        return array.reshape(1, 1, -1)
    if array.ndim == 2:
        return array.reshape(array.shape[0], 1, array.shape[1])
    if array.ndim == 3:
        return array
    return array.reshape(array.shape[0], int(np.prod(array.shape[1:-1])), array.shape[-1])


def to_fedot_input_data(
        X: Any,
        y: Any | None = None,
        *,
        task_type: str = "classification",
        use_torch: bool = False,
        torch_device: Any = "auto",
):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    features = normalize_time_series_tensor(X)
    if use_torch:
        features = _to_torch(features, device=torch_device)
    task_map = {
        "classification": Task(TaskTypesEnum.classification),
        "regression": Task(TaskTypesEnum.regression),
    }
    if task_type not in task_map:
        raise ValueError(f"Unsupported supervised task_type: {task_type}")
    target = None if y is None else np.asarray(y).reshape(-1, 1)
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=task_map[task_type],
        data_type=DataTypesEnum.image,
    )


def _features_to_fedot_input_data(features: Any, template: Any, *, use_torch: bool, torch_device: Any = "auto"):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum

    prepared = _to_torch(features, device=torch_device) if use_torch else _to_numpy(features)
    data_type = DataTypesEnum.image if len(prepared.shape) >= 3 else DataTypesEnum.table
    return InputData(
        idx=getattr(template, "idx", np.arange(prepared.shape[0])),
        features=prepared,
        target=getattr(template, "target", None),
        task=getattr(template, "task", None),
        data_type=data_type,
        supplementary_data=getattr(template, "supplementary_data", None),
    )


def _unwrap_operation_output(value: Any) -> Any:
    prediction = getattr(value, "predict", None)
    if prediction is not None and not callable(prediction):
        return prediction
    return value


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "cpu") and callable(values.cpu) and values.__class__.__module__.startswith("torch"):
        values = values.cpu().numpy()
    elif hasattr(values, "values") and not isinstance(values, np.ndarray):
        values = values.values
    return np.asarray(values)


def resolve_torch_device(device: Any = "auto"):
    import torch

    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = "auto" if device is None else str(device).strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch_device='cuda' was requested, but CUDA is not available.")
    return resolved


def _to_torch(values: Any, *, device: Any = "auto"):
    import torch

    resolved_device = resolve_torch_device(device)
    if isinstance(values, torch.Tensor):
        return values.to(device=resolved_device, dtype=torch.float32)
    return torch.as_tensor(_to_numpy(values), dtype=torch.float32, device=resolved_device)


def _operation_params(params: dict[str, Any]):
    try:
        from fedot.core.operations.operation_parameters import OperationParameters

        return OperationParameters(params)
    except Exception:
        return params


def _load_operation_class(spec: OperationSpec):
    module = import_module(spec.module_path)
    return getattr(module, spec.class_name)


def _call_transform(operation: Any, input_data: Any):
    transform = operation.transform
    parameters = inspect.signature(transform).parameters
    if "use_cache" in parameters:
        return transform(input_data, use_cache=False)
    return transform(input_data)


class KernelFeatureGeneratorMixin:
    """Default feature-generator kernel contract via features -> kernel builder."""

    def kernel(
            self,
            X_left: FeatureInput,
            X_right: FeatureInput | None = None,
            *,
            task_type: KernelTaskType | str | None = None,
    ) -> KernelBundle:
        from fedot_ind.core.kernel_learning.kernels import KernelMatrixBuilder

        resolved_task_type = task_type or getattr(self, "task_type_", None) or "classification"
        train_bundle = self.fit_transform(X_left, None, task_type=resolved_task_type)
        builder = KernelMatrixBuilder()
        kernel_bundle = builder.fit_transform(
            train_bundle.features,
            name=train_bundle.name,
            train_features=train_bundle.features,
        )
        if X_right is None:
            return kernel_bundle
        test_bundle = self.transform(X_right)
        return builder.build_test_bundle(test_bundle.features, kernel_bundle)


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
        self.operations_ = [self._build_operation(spec) for spec in self.operation_specs]
        self.train_features_ = self._run_operations(X, y)
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        if not self.operations_:
            raise ValueError(f"Feature generator {self.name!r} must be fitted before transform.")
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
        operation_cls = _load_operation_class(spec)
        return operation_cls(_operation_params(dict(spec.params)))

    def _run_operations(self, X: Any, y: Any | None) -> np.ndarray:
        if not self.operation_specs:
            raise ValueError(f"Feature generator {self.name!r} has no operation specs.")

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
            output = _call_transform(operation, current)
            raw_output = _unwrap_operation_output(output)
            next_index = index + 1
            if next_index < len(self.operation_specs):
                current = _features_to_fedot_input_data(
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
        if self.resolved_torch_device_ is not None:
            diagnostics["torch_device"] = self.resolved_torch_device_
        return FeatureBundle(
            name=self.name,
            features=matrix,
            diagnostics=diagnostics,
        )


@dataclass
class BudgetedRepositoryFeatureGeneratorAdapter(RepositoryFeatureGeneratorAdapter):
    budget_policy: GeneratorBudgetPolicy = field(default_factory=GeneratorBudgetPolicy)
    fallback_generator_: Any | None = None
    budget_diagnostics_: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        allowed, diagnostics = self.budget_policy.allows(X)
        self.budget_diagnostics_ = diagnostics
        if not allowed:
            return self._fit_fallback(X, y, task_type=task_type, reason="budget_exceeded")
        try:
            return super().fit(X, y, task_type=task_type)
        except Exception as ex:
            return self._fit_fallback(X, y, task_type=task_type,
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
            diagnostics={**bundle.diagnostics, "budget": self.budget_diagnostics_},
        )

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)

    def _fit_fallback(self, X: Any, y: Any | None, *, task_type: str, reason: str):
        self.fallback_generator_ = _build_lightweight_fallback(self.budget_policy.fallback_generator)
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
        self.train_features_ = normalize_feature_matrix(_unwrap_operation_output(prediction))
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        if self.pipeline_ is None:
            raise ValueError(f"Feature generator {self.name!r} must be fitted before transform.")
        input_data = to_fedot_input_data(X, task_type=self.task_type_ or "classification")
        prediction = self.pipeline_.predict(input_data)
        features = normalize_feature_matrix(_unwrap_operation_output(prediction))
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "fedot_pipeline", "n_features": int(features.shape[1])},
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
            diagnostics={"source": "fedot_pipeline", "n_features": int(features.shape[1])},
        )


@dataclass
class IdentityFeatureGenerator(KernelFeatureGeneratorMixin):
    name: str = "identity"
    train_features_: np.ndarray | None = None

    def fit(self, X: FeatureInput, y: TargetInput | None = None, *, task_type: str = "classification"):
        del y, task_type
        self.train_features_ = normalize_feature_matrix(X)
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        features = normalize_feature_matrix(X)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "identity", "n_features": int(features.shape[1])},
        )

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: str = "classification",
    ) -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)


@dataclass
class ShapeletFeatureGenerator(KernelFeatureGeneratorMixin):
    name: str = "shapelet_extractor"
    n_shapelets: int = 8
    window_size: int | None = None
    random_state: int = 42
    shapelets_: tuple[np.ndarray, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self):
        if self.n_shapelets < 1:
            raise ValueError("n_shapelets must be at least 1.")
        if self.window_size is not None and self.window_size < 1:
            raise ValueError("window_size must be at least 1.")

    def fit(self, X: FeatureInput, y: TargetInput | None = None, *, task_type: str = "classification"):
        del y, task_type
        tensor = normalize_time_series_tensor(X)
        n_samples, _, n_timestamps = tensor.shape
        window = self._resolve_window_size(n_timestamps)
        positions = _even_positions(n_timestamps - window + 1, self.n_shapelets)
        sample_indices = _even_positions(n_samples, self.n_shapelets)
        shapelets = []
        for shapelet_idx in range(self.n_shapelets):
            sample = tensor[sample_indices[shapelet_idx % len(sample_indices)]]
            position = positions[shapelet_idx % len(positions)]
            shapelets.append(sample[:, position:position + window].copy())
        self.shapelets_ = tuple(shapelets)
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        if not self.shapelets_:
            raise ValueError(f"Feature generator {self.name!r} must be fitted before transform.")
        tensor = normalize_time_series_tensor(X)
        features = np.column_stack([
            _min_shapelet_distance(tensor, shapelet)
            for shapelet in self.shapelets_
        ])
        return FeatureBundle(
            name=self.name,
            features=normalize_feature_matrix(features),
            diagnostics={
                "source": "shapelet",
                "n_shapelets": len(self.shapelets_),
                "window_size": int(self.shapelets_[0].shape[-1]),
                "n_features": len(self.shapelets_),
            },
        )

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: str = "classification",
    ) -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)

    def _resolve_window_size(self, n_timestamps: int) -> int:
        if self.window_size is not None:
            return min(int(self.window_size), n_timestamps)
        return max(1, min(n_timestamps, n_timestamps // 4 or 1))


@dataclass
class RandomProjectionEmbeddingFeatureGenerator(KernelFeatureGeneratorMixin):
    name: str = "embedding_extractor"
    n_components: int = 16
    random_state: int = 42
    source: str = "random_projection_embedding"
    components_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")

    def fit(self, X: FeatureInput, y: TargetInput | None = None, *, task_type: str = "classification"):
        del y, task_type
        features = normalize_feature_matrix(X)
        rng = np.random.default_rng(self.random_state)
        self.components_ = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(1, features.shape[1])),
            size=(features.shape[1], self.n_components),
        )
        return self

    def transform(self, X: FeatureInput) -> FeatureBundle:
        if self.components_ is None:
            raise ValueError(f"Feature generator {self.name!r} must be fitted before transform.")
        features = normalize_feature_matrix(X)
        if features.shape[1] != self.components_.shape[0]:
            raise ValueError("Embedding transform feature dimensionality must match fitted data.")
        embedding = np.tanh(features @ self.components_)
        return FeatureBundle(
            name=self.name,
            features=embedding,
            diagnostics={
                "source": self.source,
                "n_components": int(self.n_components),
                "random_state": int(self.random_state),
                "n_features": int(embedding.shape[1]),
            },
        )

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: str = "classification",
    ) -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        return self.transform(X)


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
                _torch_quantile_spec(params or _DEFAULT_STATISTICAL_PARAMS),
            ),
            torch_device=torch_device,
        )


def build_generator_registry() -> dict[str, Callable[[], Any]]:
    registry: dict[str, Callable[[], Any]] = {
        "identity": IdentityFeatureGenerator,
        "statistical_summary": SummaryFeatureGenerator,
        "statistical": lambda: SummaryFeatureGenerator(name="statistical"),
        "shapelet_extractor": ShapeletFeatureGenerator,
        "local_pattern_extractor": lambda: ShapeletFeatureGenerator(
            name="local_pattern_extractor",
            n_shapelets=12,
            window_size=5,
        ),
        "embedding_extractor": RandomProjectionEmbeddingFeatureGenerator,
        "foundation_embedding": lambda: RandomProjectionEmbeddingFeatureGenerator(
            name="foundation_embedding",
            n_components=32,
            source="foundation_embedding_adapter",
        ),
        "quantile_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="quantile_extractor",
            operation_specs=(_torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS),),
        ),
        "quantile_extractor_torch": lambda: RepositoryFeatureGeneratorAdapter(
            name="quantile_extractor_torch",
            operation_specs=(_torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS),),
        ),
        "wavelet_basis": lambda: RepositoryFeatureGeneratorAdapter(
            name="wavelet_basis",
            operation_specs=(_wavelet_spec(),),
        ),
        "wavelet_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="wavelet_extractor",
            operation_specs=(_wavelet_spec(), _torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS)),
        ),
        "fourier_basis": lambda: RepositoryFeatureGeneratorAdapter(
            name="fourier_basis",
            operation_specs=(_fourier_spec(),),
        ),
        "fourier_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="fourier_extractor",
            operation_specs=(_fourier_spec(), _torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS)),
        ),
        "eigen_basis": lambda: RepositoryFeatureGeneratorAdapter(
            name="eigen_basis",
            operation_specs=(_eigen_spec(),),
        ),
        "eigen_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="eigen_extractor",
            operation_specs=(_eigen_spec(), _torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS)),
        ),
        "recurrence_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="recurrence_extractor",
            operation_specs=(_recurrence_spec(),),
        ),
        "topological_extractor": lambda: BudgetedRepositoryFeatureGeneratorAdapter(
            name="topological_extractor",
            operation_specs=(_topological_spec(),),
            budget_policy=GeneratorBudgetPolicy(
                max_cells=250_000,
                fallback_generator="identity",
            ),
        ),
        "tabular_extractor": lambda: BudgetedRepositoryFeatureGeneratorAdapter(
            name="tabular_extractor",
            operation_specs=(_tabular_spec(),),
            budget_policy=GeneratorBudgetPolicy(
                max_samples=25,
                max_cells=10_000,
                fallback_generator="statistical_summary",
            ),
        ),
    }
    _extend_with_legacy_pipeline_generators(registry)
    return registry


def create_feature_generator(
        name: str,
        *,
        registry: dict[str, Callable[[], Any]] | None = None,
        torch_device: Any = "auto",
):
    source = registry or build_generator_registry()
    if name not in source:
        raise ValueError(f"Unknown kernel feature generator: {name}")
    generator = source[name]()
    if isinstance(generator, RepositoryFeatureGeneratorAdapter):
        generator.torch_device = torch_device
    return generator


def resolve_generator_operation_specs(
        name: str,
        *,
        materialize_basis: bool = False,
) -> tuple[OperationSpec, ...]:
    generator = create_feature_generator(name)
    if isinstance(generator, IdentityFeatureGenerator):
        return ()
    if not isinstance(generator, RepositoryFeatureGeneratorAdapter):
        return ()

    specs = tuple(generator.operation_specs)
    if (
            materialize_basis
            and len(specs) == 1
            and specs[0].name in BASIS_ONLY_GENERATORS
    ):
        return specs + (_torch_quantile_spec(_DEFAULT_STATISTICAL_PARAMS),)
    return specs


def _build_lightweight_fallback(name: str):
    if name == "identity":
        return IdentityFeatureGenerator()
    if name in {"statistical_summary", "statistical", "quantile_extractor", "quantile_extractor_torch"}:
        return SummaryFeatureGenerator(name=name)
    if name in {"embedding_extractor", "foundation_embedding"}:
        return RandomProjectionEmbeddingFeatureGenerator(name=name)
    if name in {"shapelet_extractor", "local_pattern_extractor"}:
        return ShapeletFeatureGenerator(name=name)
    return IdentityFeatureGenerator(name=f"{name}_fallback")


def _even_positions(size: int, count: int) -> np.ndarray:
    if size <= 0:
        return np.array([0], dtype=int)
    if count <= 1:
        return np.array([0], dtype=int)
    return np.unique(np.linspace(0, size - 1, num=count, dtype=int))


def _min_shapelet_distance(tensor: np.ndarray, shapelet: np.ndarray) -> np.ndarray:
    n_samples, _, n_timestamps = tensor.shape
    window = shapelet.shape[-1]
    if n_timestamps < window:
        padded = np.pad(tensor, ((0, 0), (0, 0), (0, window - n_timestamps)), mode="edge")
        return np.linalg.norm((padded - shapelet).reshape(n_samples, -1), axis=1)
    distances = []
    for position in range(n_timestamps - window + 1):
        segment = tensor[:, :, position:position + window]
        distances.append(np.linalg.norm((segment - shapelet).reshape(n_samples, -1), axis=1))
    return np.min(np.vstack(distances), axis=0)


def _extend_with_legacy_pipeline_generators(registry: dict[str, Callable[[], Any]]) -> None:
    try:
        from fedot_ind.core.repository.constanst_repository import KERNEL_BASELINE_FEATURE_GENERATORS

        for name, pipeline_factory in KERNEL_BASELINE_FEATURE_GENERATORS.items():
            registry.setdefault(
                name,
                lambda generator_name=name, factory=pipeline_factory: PipelineFeatureGeneratorAdapter(
                    name=generator_name,
                    pipeline_factory=factory,
                ),
            )
    except Exception:
        # Keep registry construction importable without the optional FEDOT/dask stack.
        pass


_DEFAULT_STATISTICAL_PARAMS = {
    "window_size": 10,
    "use_sliding_window": True,
    "stride": 1,
    "add_global_features": True,
    "use_cache": False,
}


def _torch_quantile_spec(params: dict[str, Any]) -> OperationSpec:
    return OperationSpec(
        name="quantile_extractor_torch",
        module_path="fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor",
        class_name="TorchQuantileExtractor",
        params=params,
        use_torch=True,
    )


def _wavelet_spec() -> OperationSpec:
    return OperationSpec(
        name="wavelet_basis",
        module_path="fedot_ind.core.operation.transformation.basis.wavelet",
        class_name="WaveletBasisImplementation",
        params={"wavelet": "mexh", "n_components": 2, "use_cache": False},
    )


def _fourier_spec() -> OperationSpec:
    return OperationSpec(
        name="fourier_basis",
        module_path="fedot_ind.core.operation.transformation.basis.fourier",
        class_name="FourierBasisImplementation",
        params={
            "spectrum_type": "smoothed",
            "threshold": 0.9,
            "output_format": "signal",
            "approximation": "exact",
            "low_rank": 5,
            "use_cache": False,
        },
    )


def _eigen_spec() -> OperationSpec:
    return OperationSpec(
        name="eigen_basis",
        module_path="fedot_ind.core.operation.transformation.basis.eigen_basis",
        class_name="EigenBasisImplementation",
        params={
            "window_size": 20,
            "rank_regularization": "explained_dispersion",
            "decomposition_type": "svd",
            "use_cache": False,
        },
    )


def _recurrence_spec() -> OperationSpec:
    return OperationSpec(
        name="recurrence_extractor",
        module_path="fedot_ind.core.operation.transformation.representation.recurrence.recurrence_extractor",
        class_name="RecurrenceExtractor",
        params={
            "window_size": 10,
            "stride": 1,
            "rec_metric": "cosine",
            "use_sliding_window": False,
            "image_mode": False,
            "use_cache": False,
        },
    )


def _topological_spec() -> OperationSpec:
    return OperationSpec(
        name="topological_extractor",
        module_path="fedot_ind.core.operation.transformation.representation.topological.topological_extractor",
        class_name="TopologicalExtractor",
        params={"window_size": 10, "stride": 1, "use_cache": False},
    )


def _tabular_spec() -> OperationSpec:
    return OperationSpec(
        name="tabular_extractor",
        module_path="fedot_ind.core.operation.transformation.representation.tabular.tabular_extractor",
        class_name="TabularExtractor",
        params={"feature_domain": "all", "reduce_dimension": True, "use_cache": False},
    )

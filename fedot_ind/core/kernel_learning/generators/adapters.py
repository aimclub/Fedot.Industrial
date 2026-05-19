from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Sequence

import numpy as np

from fedot_ind.core.kernel_learning.contracts import FeatureBundle

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


@dataclass
class RepositoryFeatureGeneratorAdapter:
    name: str
    operation_specs: Sequence[OperationSpec]
    torch_device: Any = "auto"
    task_type_: str | None = None
    operations_: list[Any] = field(default_factory=list)
    train_features_: np.ndarray | None = None
    resolved_torch_device_: str | None = None

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        self.task_type_ = task_type
        self.operations_ = [self._build_operation(spec) for spec in self.operation_specs]
        self.train_features_ = self._run_operations(X, y)
        return self

    def transform(self, X: Any) -> FeatureBundle:
        if not self.operations_:
            raise ValueError(f"Feature generator {self.name!r} must be fitted before transform.")
        features = self._run_operations(X, None)
        return self._bundle(features)

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
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
class PipelineFeatureGeneratorAdapter:
    name: str
    pipeline_factory: Any
    task_type_: str | None = None
    pipeline_: Any | None = None
    train_features_: np.ndarray | None = None

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        self.task_type_ = task_type
        self.pipeline_ = deepcopy(self.pipeline_factory).build()
        input_data = to_fedot_input_data(X, y, task_type=task_type)
        self.pipeline_.fit(input_data)
        prediction = self.pipeline_.predict(input_data)
        self.train_features_ = normalize_feature_matrix(_unwrap_operation_output(prediction))
        return self

    def transform(self, X: Any) -> FeatureBundle:
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

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
        self.fit(X, y, task_type=task_type)
        features = normalize_feature_matrix(self.train_features_)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "fedot_pipeline", "n_features": int(features.shape[1])},
        )


@dataclass
class IdentityFeatureGenerator:
    name: str = "identity"
    train_features_: np.ndarray | None = None

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        del y, task_type
        self.train_features_ = normalize_feature_matrix(X)
        return self

    def transform(self, X: Any) -> FeatureBundle:
        features = normalize_feature_matrix(X)
        return FeatureBundle(
            name=self.name,
            features=features,
            diagnostics={"source": "identity", "n_features": int(features.shape[1])},
        )

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
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
        "topological_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="topological_extractor",
            operation_specs=(_topological_spec(),),
        ),
        "tabular_extractor": lambda: RepositoryFeatureGeneratorAdapter(
            name="tabular_extractor",
            operation_specs=(_tabular_spec(),),
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

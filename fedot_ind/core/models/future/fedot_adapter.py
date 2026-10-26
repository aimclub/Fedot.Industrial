"""Array-interface adapters for the existing FUTURE multimodal classifier."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
from inspect import Parameter, signature
from math import isfinite
from numbers import Integral, Real

import numpy as np
import torch

from fedot_ind.core.models.future.future_clf import ConfigurableMultimodalFusionClassifier
from fedot_ind.core.models.future.mapping import FUSION_REGISTRY
from fedot_ind.core.models.future.rules import (
    validate_context_modalities_for_raw_centered,
    validate_supported_fusion_method,
)
from fedot_ind.core.models.future.tools import FutureTrainingConfig
from fedot_ind.core.models.future.trainer import FutureClassifierTrainer
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn
from fedot_ind.core.multimodal.batching import make_bundle_dataloader
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.mapping import MODALITY_CAPABILITIES
from fedot_ind.core.multimodal.packing import (
    parse_layout,
    unpack_modalities,
    validate_packed_values,
)
from fedot_ind.core.operation.transformation.torch_backend.io import (
    resolve_torch_device,
    set_torch_seed,
)


def _integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")


def _real(name: str, value: float, minimum: float | None = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _options(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping with string keys.")
    return deepcopy(dict(value))


def _optional_parameters(callable_) -> dict:
    return {
        name: parameter.default
        for name, parameter in signature(callable_).parameters.items()
        if parameter.default is not Parameter.empty
    }


def _validate_options(options: dict, defaults: dict, name: str) -> None:
    """Check the existing constructors' option schema without creating weights."""

    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}.")
    for key, value in options.items():
        label = f"{name}.{key}"
        default = defaults[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise TypeError(f"{label} must be bool.")
        elif isinstance(default, int):
            _integer(label, value)
        elif isinstance(default, float):
            _real(label, value, minimum=None if key in ("gamma_scale", "beta_scale") else 0.0)
            if key == "dropout" and value > 1:
                raise ValueError(f"{label} must be in [0, 1].")
            if key == "mlp_ratio" and value == 0:
                raise ValueError(f"{label} must be positive.")
        elif isinstance(default, tuple):
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{label} must be a list or tuple of positive integers.")
            for size in value:
                _integer(label, size)
        elif isinstance(default, str) and not isinstance(value, str):
            raise TypeError(f"{label} must be a string.")
        if key == "activation":
            try:
                get_activation_fn(value)
            except KeyError as exc:
                raise ValueError(f"Unsupported {label}: {value!r}.") from exc
        if key == "pooling" and value not in ("mean", "cls", "concat"):
            raise ValueError(f"{label} must be 'mean', 'cls', or 'concat'.")


def _training_config(payload) -> FutureTrainingConfig:
    if isinstance(payload, FutureTrainingConfig):
        config = deepcopy(payload)
    else:
        options = _options(payload, "training_config")
        unknown = set(options) - {field.name for field in fields(FutureTrainingConfig)}
        if unknown:
            raise ValueError(f"Unknown training_config options: {sorted(unknown)}.")
        config = FutureTrainingConfig(**options)
    _integer("training_config.epochs", config.epochs)
    _integer("training_config.batch_size", config.batch_size)
    _real("training_config.learning_rate", config.learning_rate)
    if config.learning_rate == 0:
        raise ValueError("training_config.learning_rate must be positive.")
    _real("training_config.weight_decay", config.weight_decay)
    if config.early_stopping_patience is not None:
        _integer("training_config.early_stopping_patience", config.early_stopping_patience)
    if config.seed is not None:
        _integer("training_config.seed", config.seed, minimum=0)
        if config.seed >= 2**64:
            raise ValueError("training_config.seed must be less than 2**64.")
    if not isinstance(config.timing_warmup, bool):
        raise TypeError("training_config.timing_warmup must be bool.")
    if not isinstance(config.device, (str, torch.device)):
        raise TypeError("training_config.device must be a string or torch.device.")
    config.device = resolve_torch_device(config.device)
    return config


class ModalitySlice:
    """Select one modality's packed columns, retaining the array input type."""

    def __init__(self, input_width: int, start: int, stop: int):
        _integer("input_width", input_width)
        _integer("start", start, minimum=0)
        _integer("stop", stop)
        if not start < stop <= input_width:
            raise ValueError("Slice bounds must satisfy 0 <= start < stop <= input_width.")
        self.input_width = input_width
        self.start = start
        self.stop = stop

    def fit(self, X, y=None) -> "ModalitySlice":
        validate_packed_values(X, self.input_width)
        return self

    def transform(self, X) -> np.ndarray | torch.Tensor:
        validate_packed_values(X, self.input_width)
        selected = X[:, self.start:self.stop]
        if isinstance(selected, torch.Tensor):
            return selected.detach().clone()
        return selected.copy()


class PackedFusionClassifier:
    """Restore shaped modalities and delegate training to FutureClassifierTrainer.

    Class vocabulary is learned only from ``fit`` targets. A failed fit, including
    a failed refit, discards all fitted state. Empty batches are rejected.
    """

    def __init__(
        self,
        modality_layout,
        fusion_method="concat",
        d_model=16,
        encoder_kwargs=None,
        fusion_kwargs=None,
        training_config=None,
        head_dropout=0.0,
        head_hidden_dim=None,
    ):
        self.layout = parse_layout(modality_layout)
        self.modality_layout = self.layout
        self.input_width = sum(item.width for item in self.layout)
        self.fusion_method = validate_supported_fusion_method(fusion_method, FUSION_REGISTRY)
        _integer("d_model", d_model)
        _real("head_dropout", head_dropout)
        if head_dropout > 1:
            raise ValueError("head_dropout must be in [0, 1].")
        if head_hidden_dim is not None:
            _integer("head_hidden_dim", head_hidden_dim)
        self.d_model = d_model
        self.head_dropout = head_dropout
        self.head_hidden_dim = head_hidden_dim
        self.encoder_kwargs = _options(encoder_kwargs, "encoder_kwargs")
        self.fusion_kwargs = _options(fusion_kwargs, "fusion_kwargs")
        self.training_config = _training_config(training_config)
        self._validate_model_config()
        self._clear_fitted_state()

    def _validate_model_config(self) -> None:
        modalities = tuple(MultimodalModality(item.name) for item in self.layout)
        entry = FUSION_REGISTRY[self.fusion_method]
        if entry.requires_raw:
            validate_context_modalities_for_raw_centered(MultimodalModality.raw, modalities)
        unknown = set(self.encoder_kwargs) - {item.name for item in self.layout}
        if unknown:
            raise ValueError(f"encoder_kwargs refers to absent modalities: {sorted(unknown)}.")
        for item in self.layout:
            raw_options = self.encoder_kwargs.get(item.name, {})
            if raw_options is None:
                raise TypeError(f"encoder_kwargs.{item.name} must be a mapping, not None.")
            options = _options(raw_options, f"encoder_kwargs.{item.name}")
            preset = MODALITY_CAPABILITIES[MultimodalModality(item.name)].encoder_preset
            defaults = _optional_parameters(preset.builder)
            defaults.pop("d_model", None)
            defaults.pop(preset.shape_arg_name.value, None)
            _validate_options(options, defaults, f"encoder_kwargs.{item.name}")
            preset.build_config(shape=(1, *item.shape), d_model=self.d_model, kwargs=options)
            self.encoder_kwargs[item.name] = options
        # Bottleneck subclasses forward options to their shared base constructor.
        defaults = {}
        for cls in reversed(entry.fusion_class.__mro__):
            if cls is not torch.nn.Module and "__init__" in cls.__dict__:
                defaults.update(_optional_parameters(cls.__init__))
        defaults.pop("d_model", None)
        _validate_options(self.fusion_kwargs, defaults, "fusion_kwargs")
        if "num_heads" in defaults:
            heads = self.fusion_kwargs.get("num_heads", defaults["num_heads"])
            if self.d_model % heads:
                raise ValueError("d_model must be divisible by fusion_kwargs.num_heads.")

    def _clear_fitted_state(self) -> None:
        self.model_ = None
        self.trainer_ = None
        self.classes_ = None

    def fit(self, X, y) -> "PackedFusionClassifier":
        self._clear_fitted_state()
        bundle = unpack_modalities(X, self.layout)
        if isinstance(y, torch.Tensor):
            if y.layout != torch.strided or y.device.type == "meta" or y.is_quantized:
                raise TypeError("y must be a materialized, unquantized strided tensor.")
            if y.dtype == torch.bool or y.is_complex():
                raise TypeError("y must contain real numeric class labels, not bool or complex.")
            labels_tensor = y.detach().cpu()
            if labels_tensor.is_floating_point():
                labels_tensor = labels_tensor.to(dtype=torch.float64)
            labels = labels_tensor.numpy().copy()
        else:
            labels = np.array(y, copy=True)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1 or labels.shape[0] != bundle.n_samples:
            raise ValueError("y must have one class label per sample, with shape (samples,) or (samples, 1).")
        if labels.dtype.kind not in "iuf":
            raise TypeError("y must contain real numeric class labels, not bool or object.")
        if not np.isfinite(labels).all():
            raise ValueError("y must contain only finite class labels.")
        classes, encoded = np.unique(labels, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Training targets must contain at least two unique classes.")
        bundle = bundle.with_target(torch.as_tensor(encoded, dtype=torch.long, device=bundle.device).clone())
        config = deepcopy(self.training_config)
        set_torch_seed(config.seed)
        model = ConfigurableMultimodalFusionClassifier(
            num_classes=len(classes),
            fusion_method=self.fusion_method,
            d_model=self.d_model,
            modalities=tuple(item.name for item in self.layout),
            encoder_kwargs=deepcopy(self.encoder_kwargs),
            fusion_kwargs=deepcopy(self.fusion_kwargs),
            head_dropout=self.head_dropout,
            head_hidden_dim=self.head_hidden_dim,
        )
        trainer = FutureClassifierTrainer(model=model, config=config)
        loader = make_bundle_dataloader(
            bundle,
            batch_size=config.batch_size,
            shuffle=True,
            device=trainer.device,
            seed=config.seed,
            require_target=True,
        )
        trainer.fit(loader, build_bundle=bundle)
        if trainer.history is None or not all(isfinite(loss) for loss in trainer.history.train_loss):
            raise ValueError("Training did not produce finite losses.")
        self.model_, self.trainer_, self.classes_ = model, trainer, classes.copy()
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.trainer_ is None:
            raise ValueError("PackedFusionClassifier is not fitted; call fit before prediction.")
        bundle = unpack_modalities(X, self.layout)
        probabilities = self.trainer_.predict_proba(bundle).detach().cpu().numpy().copy()
        if probabilities.shape != (bundle.n_samples, len(self.classes_)) or not np.isfinite(probabilities).all():
            raise ValueError("Classifier probabilities must be a finite sample-by-class matrix.")
        return probabilities

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)].copy()

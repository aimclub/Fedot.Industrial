from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from benchmark.industrial.core import ModelSpec, RunStatus
from benchmark.industrial.errors import BenchmarkClassificationError
from benchmark.industrial.models.kernel_artifacts import export_kernel_learning_artifacts


@dataclass
class MajorityClassClassifier:
    name: str = 'MajorityClass'
    tags: tuple[str, ...] = ('baseline', 'classification')
    optional: bool = False
    majority_label_: str = ''

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        del features
        values, counts = np.unique(target, return_counts=True)
        self.majority_label_ = str(values[np.argmax(counts)])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.majority_label_, dtype=object)


@dataclass
class NearestCentroidClassifier:
    name: str = 'NearestCentroid'
    tags: tuple[str, ...] = ('baseline', 'classification')
    optional: bool = False
    centroids_: dict[str, np.ndarray] | None = None

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self.centroids_ = {}
        for label in np.unique(target):
            self.centroids_[str(label)] = np.mean(features[target == label], axis=0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise BenchmarkClassificationError('NearestCentroidClassifier must be fitted before prediction.')
        predictions = []
        for row in features:
            label = min(
                self.centroids_.items(),
                key=lambda item: float(np.linalg.norm(row - item[1])),
            )[0]
            predictions.append(label)
        return np.asarray(predictions, dtype=object)


@dataclass
class OptionalExternalClassifier:
    dependency_name: str
    name: str
    tags: tuple[str, ...] = ('industrial', 'classification', 'external')
    optional: bool = True

    def availability(self) -> tuple[RunStatus, str]:
        try:
            __import__(self.dependency_name)
            return RunStatus.SKIPPED, 'Adapter scaffold registered but training backend is not wired yet.'
        except Exception:
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'


@dataclass
class KernelEnsembleClassifierAdapter:
    name: str
    tags: tuple[str, ...] = ('industrial', 'classification', 'kernel_learning')
    optional: bool = False
    params: dict[str, Any] | None = None
    model_: Any | None = None

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from fedot_ind.core.kernel_learning import KernelEnsembleClassifier  # noqa: F401
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover
            return RunStatus.NOT_AVAILABLE, f'Kernel ensemble classifier is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from fedot_ind.core.kernel_learning import KernelEnsembleClassifier

        self.model_ = KernelEnsembleClassifier(**(self.params or {}))
        self.model_.fit(features, target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise BenchmarkClassificationError('KernelEnsembleClassifierAdapter must be fitted before prediction.')
        return self.model_.predict(features)

    def export_artifacts(self) -> dict[str, Any]:
        return export_kernel_learning_artifacts(self.model_)


@dataclass
class PDLClassifierAdapter:
    name: str
    tags: tuple[str, ...] = ('industrial', 'classification', 'pdl')
    optional: bool = True
    params: dict[str, Any] | None = None
    model_: Any | None = None

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from fedot.core.data.data import InputData  # noqa: F401
            from fedot.core.operations.operation_parameters import OperationParameters  # noqa: F401
            from fedot.core.repository.dataset_types import DataTypesEnum  # noqa: F401
            from fedot.core.repository.tasks import Task, TaskTypesEnum  # noqa: F401
            from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceClassifier  # noqa: F401
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover - optional FEDOT runtime boundary
            return RunStatus.NOT_AVAILABLE, f'PDL classifier is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceClassifier

        input_data = _fedot_input_data(features=features, target=target, task_type='classification')
        self.model_ = PairwiseDifferenceClassifier(params=_operation_parameters(self.params, default_model='rf'))
        self.model_.fit(input_data)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise BenchmarkClassificationError('PDLClassifierAdapter must be fitted before prediction.')
        dummy_target = np.zeros(features.shape[0], dtype=int)
        input_data = _fedot_input_data(features=features, target=dummy_target, task_type='classification')
        prediction = self.model_.predict(input_data)
        values = getattr(prediction, 'predict', prediction)
        return np.asarray(values).reshape(-1).astype(object)


@dataclass
class MiniRocketRidgeClassifierAdapter:
    """MiniRocket feature transform + RidgeClassifierCV baseline for TSC benchmarks.

    Uses the same MiniRocketFeatures / get_minirocket_features primitives as
    MiniRocketExtractor, but fits kernels once on train and reuses them on test.
    """

    name: str
    tags: tuple[str, ...] = ('baseline', 'classification', 'minirocket')
    optional: bool = False
    params: dict[str, Any] | None = None
    kernels_: list[Any] | None = None
    classifier_: Any | None = None
    num_features_: int | None = None
    random_state_: int | None = None

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from sklearn.linear_model import RidgeClassifierCV  # noqa: F401

            from fedot_ind.core.models.nn.network_impl.feature_extraction.mini_rocket import (  # noqa: F401
                MiniRocketFeatures,
                get_minirocket_features,
            )
            from fedot_ind.core.operation.transformation.torch_backend.io import (  # noqa: F401
                normalize_time_series_tensor,
            )
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover
            return RunStatus.NOT_AVAILABLE, f'MiniRocketRidge classifier is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from sklearn.linear_model import RidgeClassifierCV

        params = dict(self.params or {})
        num_features = int(params.pop('num_features', 1000))
        random_state = params.pop('random_state', params.pop('seed', 42))
        if random_state is not None:
            random_state = int(random_state)
        max_dilations_per_kernel = int(params.pop('max_dilations_per_kernel', 32))
        chunksize = params.pop('chunksize', None)
        device = params.pop('device', 'cpu')
        ridge_kwargs = dict(params.pop('ridge', {}))
        if params:
            raise BenchmarkClassificationError(
                f'Unsupported MiniRocketRidge adapter params: {sorted(params)}'
            )

        series = _to_minirocket_series(features)
        n_channels = int(series.shape[1])
        self.random_state_ = random_state
        self.kernels_ = []

        channel_features: list[np.ndarray] = []
        for channel_index in range(n_channels):
            channel = series[:, channel_index : channel_index + 1, :]
            kernel = _fit_minirocket_kernel(
                channel,
                num_features=num_features,
                max_dilations_per_kernel=max_dilations_per_kernel,
                random_state=random_state,
                device=device,
            )
            self.kernels_.append(kernel)
            channel_features.append(
                _extract_minirocket_features(channel, kernel, chunksize=chunksize)
            )

        feature_matrix = np.concatenate(channel_features, axis=1)
        self.num_features_ = int(feature_matrix.shape[1])
        self.classifier_ = RidgeClassifierCV(**ridge_kwargs)
        self.classifier_.fit(feature_matrix, np.asarray(target).reshape(-1))

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.kernels_ is None or self.classifier_ is None:
            raise BenchmarkClassificationError(
                'MiniRocketRidgeClassifierAdapter must be fitted before prediction.'
            )
        feature_matrix = self._transform_features(features)
        return np.asarray(self.classifier_.predict(feature_matrix), dtype=object)

    def export_artifacts(self) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            'adapter': 'minirocket_ridge_classifier',
            'num_features': self.num_features_,
            'random_state': self.random_state_,
            'n_kernels': None if self.kernels_ is None else len(self.kernels_),
        }
        if self.classifier_ is not None:
            artifacts['ridge'] = {
                'alpha_': _json_safe(getattr(self.classifier_, 'alpha_', None)),
                'coef_shape': list(getattr(self.classifier_, 'coef_', np.empty(0)).shape),
                'classes_': [
                    _json_safe(label)
                    for label in getattr(self.classifier_, 'classes_', ())
                ],
            }
        return artifacts

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        assert self.kernels_ is not None
        params = dict(self.params or {})
        chunksize = params.get('chunksize')
        series = _to_minirocket_series(features)
        if series.shape[1] != len(self.kernels_):
            raise BenchmarkClassificationError(
                f'MiniRocketRidge expected {len(self.kernels_)} channels, '
                f'got {series.shape[1]}.'
            )
        channel_features = [
            _extract_minirocket_features(
                series[:, channel_index : channel_index + 1, :],
                kernel,
                chunksize=chunksize,
            )
            for channel_index, kernel in enumerate(self.kernels_)
        ]
        return np.concatenate(channel_features, axis=1)


@dataclass
class FutureMultimodalClassifierAdapter:
    """Thin benchmark adapter over MultimodalDatasetPreparer + FutureClassifierTrainer.

    Supported top-level params include modalities, fusion_method, d_model, and
    nested ``preparation`` / ``training`` blocks. Training accepts seed, device,
    epochs, learning_rate, patience (alias of early_stopping_patience),
    batch_size or train_batch_size/val_batch_size, and validation_fraction as the
    train/val split policy. Set ``output_diagnostics=True`` to attach compact
    fusion diagnostics to export_artifacts().
    """

    name: str
    tags: tuple[str, ...] = ('industrial', 'classification', 'future', 'multimodal')
    optional: bool = True
    params: dict[str, Any] | None = None
    preparer_: Any | None = None
    trainer_: Any | None = None
    label_mapping_: dict[str, int] | None = None
    inverse_label_mapping_: dict[int, str] | None = None
    history_: dict[str, Any] | None = None
    diagnostics_: dict[str, Any] | None = None
    output_diagnostics_: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        try:
            from fedot_ind.core.models.future import (  # noqa: F401
                ConfigurableMultimodalFusionClassifier,
                FutureClassifierTrainer,
            )
            from fedot_ind.core.multimodal.preparation import (  # noqa: F401
                MultimodalDatasetPreparer,
            )
            return RunStatus.SUCCESS, 'ready'
        except Exception as exc:  # pragma: no cover
            return RunStatus.NOT_AVAILABLE, f'FUTURE multimodal classifier is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from fedot_ind.core.models.future import (
            ConfigurableMultimodalFusionClassifier,
            FutureClassifierTrainer,
            FutureTrainingConfig,
        )
        from fedot_ind.core.multimodal.configs import build_preparation_config
        from fedot_ind.core.multimodal.preparation import MultimodalDatasetPreparer

        params = dict(self.params or {})
        preparation_kwargs = dict(params.pop('preparation', {}))
        training_kwargs = dict(params.pop('training', {}))
        self.output_diagnostics_ = bool(params.pop('output_diagnostics', False))
        classifier_kwargs = dict(params)

        fusion_method = classifier_kwargs.pop('fusion_method', 'concat')
        d_model = int(classifier_kwargs.pop('d_model', 64))
        modalities = classifier_kwargs.pop('modalities', None)
        fusion_kwargs = dict(classifier_kwargs.pop('fusion_kwargs', {}))
        raw_modality = classifier_kwargs.pop('raw_modality', 'raw')
        encoder_kwargs = dict(classifier_kwargs.pop('encoder_kwargs', {}))
        head_hidden_dim = classifier_kwargs.pop('head_hidden_dim', None)
        head_dropout = float(classifier_kwargs.pop('head_dropout', 0.2))
        head_activation = classifier_kwargs.pop('head_activation', 'GELU')
        if classifier_kwargs:
            raise BenchmarkClassificationError(
                f'Unsupported FUTURE adapter params: {sorted(classifier_kwargs)}'
            )

        self.preparer_ = MultimodalDatasetPreparer(
            config=build_preparation_config(**preparation_kwargs)
            if preparation_kwargs
            else build_preparation_config()
        )
        train_bundle = self.preparer_.fit_transform(features, target)
        if train_bundle.target is None:
            raise BenchmarkClassificationError('Prepared train bundle is missing targets.')

        if self.preparer_.label_mapping_ is not None:
            self.label_mapping_ = {
                str(label): int(index)
                for label, index in self.preparer_.label_mapping_.items()
            }
            num_classes = len(self.label_mapping_)
        else:
            unique_targets = sorted(
                int(value) for value in train_bundle.target.unique().tolist()
            )
            if any(label < 0 for label in unique_targets):
                raise BenchmarkClassificationError(
                    'FUTURE adapter expects non-negative integer class labels.'
                )
            num_classes = int(train_bundle.target.max().item()) + 1
            self.label_mapping_ = {
                str(label): int(label) for label in unique_targets
            }
        self.inverse_label_mapping_ = {
            index: label for label, index in self.label_mapping_.items()
        }

        model = ConfigurableMultimodalFusionClassifier(
            num_classes=num_classes,
            fusion_method=fusion_method,
            d_model=d_model,
            modalities=modalities,
            encoder_kwargs=encoder_kwargs,
            fusion_kwargs=fusion_kwargs,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            head_activation=head_activation,
            raw_modality=raw_modality,
        )

        train_batch_size, val_batch_size, training_kwargs = _resolve_future_batch_sizes(
            training_kwargs
        )
        patience = training_kwargs.pop('patience', None)
        early_stopping_patience = training_kwargs.pop('early_stopping_patience', patience)
        if early_stopping_patience is not None:
            early_stopping_patience = int(early_stopping_patience)

        training_config = FutureTrainingConfig(
            epochs=int(training_kwargs.pop('epochs', 2)),
            batch_size=train_batch_size,
            learning_rate=float(training_kwargs.pop('learning_rate', 1e-3)),
            weight_decay=float(training_kwargs.pop('weight_decay', 0.0)),
            early_stopping_patience=early_stopping_patience,
            device=training_kwargs.pop('device', 'cpu'),
            seed=training_kwargs.pop('seed', 42),
        )
        # validation_fraction is the train/val split policy for the adapter.
        validation_fraction = float(training_kwargs.pop('validation_fraction', 0.0))
        drop_last = bool(training_kwargs.pop('drop_last', False))
        if training_kwargs:
            raise BenchmarkClassificationError(
                f'Unsupported FUTURE training params: {sorted(training_kwargs)}'
            )

        from fedot_ind.core.multimodal.batching import (
            make_bundle_dataloader,
            split_bundle_by_fraction,
        )

        fit_bundle = train_bundle
        val_bundle = None
        if validation_fraction > 0.0:
            fit_bundle, val_bundle = split_bundle_by_fraction(
                train_bundle,
                validation_fraction=validation_fraction,
                seed=training_config.seed,
            )

        train_loader = make_bundle_dataloader(
            fit_bundle,
            batch_size=train_batch_size,
            shuffle=True,
            device=training_config.device,
            seed=training_config.seed,
            drop_last=drop_last,
            require_target=True,
        )
        val_loader = None
        if val_bundle is not None:
            val_loader = make_bundle_dataloader(
                val_bundle,
                batch_size=val_batch_size,
                shuffle=False,
                device=training_config.device,
                seed=training_config.seed,
                drop_last=False,
                require_target=True,
            )

        self.trainer_ = FutureClassifierTrainer(model=model, config=training_config)
        history = self.trainer_.fit(
            train_loader,
            val_loader,
            build_bundle=fit_bundle,
        )
        self.history_ = {
            'train_duration_s': history.train_duration_s,
            'best_epoch': history.best_epoch,
            'stopped_early': history.stopped_early,
            'train_loss': list(history.train_loss),
            'validation_loss': list(history.validation_loss),
            'num_parameters': history.num_parameters,
            'best_validation_loss': history.best_validation_loss,
        }
        self.diagnostics_ = None
        if self.output_diagnostics_:
            aux = self.trainer_.evaluate_diagnostics(fit_bundle.without_target())
            self.diagnostics_ = _compact_fusion_diagnostics(aux)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if (
            self.preparer_ is None
            or self.trainer_ is None
            or self.inverse_label_mapping_ is None
        ):
            raise BenchmarkClassificationError(
                'FutureMultimodalClassifierAdapter must be fitted before prediction.'
            )
        test_bundle = self.preparer_.transform(features)
        predictions = (
            self.trainer_.predict(test_bundle.without_target()).detach().cpu().numpy()
        )
        return np.asarray(
            [self.inverse_label_mapping_[int(index)] for index in predictions],
            dtype=object,
        )

    def export_artifacts(self) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            'adapter': 'future_multimodal_classifier',
            'label_mapping': dict(self.label_mapping_ or {}),
            'output_diagnostics': self.output_diagnostics_,
        }
        if self.history_ is not None:
            artifacts['training_history'] = dict(self.history_)
            artifacts['train_duration_s'] = self.history_.get('train_duration_s')
        if self.diagnostics_ is not None:
            artifacts['diagnostics'] = dict(self.diagnostics_)
        if self.trainer_ is not None:
            artifacts['fusion_method'] = self.trainer_.model.fusion_method.value
            artifacts['d_model'] = self.trainer_.model.d_model
            artifacts['num_parameters'] = {
                'total': int(
                    sum(
                        parameter.numel()
                        for parameter in self.trainer_.model.parameters()
                    )
                ),
            }
        return artifacts


CLASSIFICATION_ADAPTER_REGISTRY: dict[str, type] = {
    'majority_class': MajorityClassClassifier,
    'nearest_centroid': NearestCentroidClassifier,
    'kernel_ensemble_classifier': KernelEnsembleClassifierAdapter,
    'pdl_classifier': PDLClassifierAdapter,
    'pdl_clf': PDLClassifierAdapter,
    'minirocket_ridge_classifier': MiniRocketRidgeClassifierAdapter,
    'future_multimodal_classifier': FutureMultimodalClassifierAdapter,
    'fedot_industrial_classifier': OptionalExternalClassifier,
}

_ADAPTER_EXTRA_KWARGS: dict[str, dict[str, Any]] = {
    'fedot_industrial_classifier': {'dependency_name': 'fedot'},
    'pdl_classifier': {'optional': True},
    'pdl_clf': {'optional': True},
}


def build_classification_model(spec: ModelSpec):
    key = spec.adapter_name.lower()
    adapter_cls = CLASSIFICATION_ADAPTER_REGISTRY.get(key)
    if adapter_cls is None:
        available = sorted(CLASSIFICATION_ADAPTER_REGISTRY)
        raise BenchmarkClassificationError(
            f'Unsupported classification model adapter: {spec.adapter_name}. '
            f'Available adapters: {available}.'
        )

    field_names = {item.name for item in fields(adapter_cls)}
    kwargs: dict[str, Any] = {}
    if 'name' in field_names:
        kwargs['name'] = spec.display_name
    if 'tags' in field_names and spec.tags:
        kwargs['tags'] = spec.tags
    if 'optional' in field_names:
        kwargs['optional'] = spec.optional
    if 'params' in field_names:
        kwargs['params'] = dict(spec.params)
    kwargs.update(_ADAPTER_EXTRA_KWARGS.get(key, {}))
    return adapter_cls(**kwargs)


def _operation_parameters(params: dict[str, Any] | None, *, default_model: str):
    from fedot.core.operations.operation_parameters import OperationParameters

    payload = {'model': default_model}
    payload.update(dict(params or {}))
    return OperationParameters(payload)


def _fedot_input_data(features: np.ndarray, target: np.ndarray, *, task_type: str):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    task = Task(
        TaskTypesEnum.classification
        if task_type == 'classification'
        else TaskTypesEnum.regression
    )
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=task,
        data_type=DataTypesEnum.table,
    )


def _resolve_future_batch_sizes(training_kwargs: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    kwargs = dict(training_kwargs)
    default_batch_size = int(kwargs.pop('batch_size', 32))
    train_batch_size = int(kwargs.pop('train_batch_size', default_batch_size))
    val_batch_size = int(kwargs.pop('val_batch_size', default_batch_size))
    return train_batch_size, val_batch_size, kwargs


def _to_minirocket_series(features: np.ndarray) -> np.ndarray:
    from fedot_ind.core.operation.transformation.torch_backend.io import (
        normalize_time_series_tensor,
    )

    series = normalize_time_series_tensor(features)
    return np.asarray(series, dtype=np.float32)


def _fit_minirocket_kernel(
    channel: np.ndarray,
    *,
    num_features: int,
    max_dilations_per_kernel: int,
    random_state: int | None,
    device: Any,
):
    import torch

    from fedot_ind.core.models.nn.network_impl.feature_extraction.mini_rocket import (
        MiniRocketFeatures,
    )

    kernel = MiniRocketFeatures(
        input_dim=1,
        seq_len=int(channel.shape[2]),
        num_features=num_features,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
    ).to(device)
    kernel.fit(channel)
    kernel.eval()
    return kernel


def _extract_minirocket_features(
    channel: np.ndarray,
    kernel: Any,
    *,
    chunksize: int | None,
) -> np.ndarray:
    from fedot_ind.core.models.nn.network_impl.feature_extraction.mini_rocket import (
        get_minirocket_features,
    )

    kwargs: dict[str, Any] = {'convert_to_numpy': True}
    if chunksize is not None:
        kwargs['chunksize'] = int(chunksize)
    features = get_minirocket_features(channel, kernel, **kwargs)
    # get_minirocket_features returns (N, F, 1); flatten to (N, F).
    return np.asarray(features, dtype=np.float32).reshape(features.shape[0], -1)


def _compact_fusion_diagnostics(aux: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'active_modalities': list(getattr(aux, 'active_modalities', []) or []),
        'embedding_dim': int(getattr(aux, 'embedding_dim', 0) or 0),
    }
    num_parameters = getattr(aux, 'num_parameters', None)
    if num_parameters is not None:
        payload['num_parameters'] = _json_safe(num_parameters)
    for key in (
        'alpha_stats',
        'gamma_beta_summary',
        'attention_summary',
        'pooling',
        'num_latents',
        'num_heads',
        'num_layers',
    ):
        value = getattr(aux, key, None)
        if value is not None:
            payload[key] = _json_safe(value)
    logits = getattr(aux, 'logits', None)
    if logits is not None and hasattr(logits, 'shape'):
        payload['logits_shape'] = list(logits.shape)
    h_final = getattr(aux, 'h_final', None)
    if h_final is not None and hasattr(h_final, 'shape'):
        payload['h_final_shape'] = list(h_final.shape)
    return payload


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        return _json_safe(value.detach().cpu().numpy())
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


__all__ = [
    "CLASSIFICATION_ADAPTER_REGISTRY",
    "FutureMultimodalClassifierAdapter",
    "KernelEnsembleClassifierAdapter",
    "MajorityClassClassifier",
    "MiniRocketRidgeClassifierAdapter",
    "NearestCentroidClassifier",
    "OptionalExternalClassifier",
    "PDLClassifierAdapter",
    "build_classification_model",
]

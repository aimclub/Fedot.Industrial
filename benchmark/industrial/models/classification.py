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
class FutureFusionClassifierAdapter:
    """Thin benchmark adapter over MultimodalDatasetPreparer + FutureClassifierTrainer."""

    name: str
    tags: tuple[str, ...] = ('industrial', 'classification', 'future', 'multimodal')
    optional: bool = True
    params: dict[str, Any] | None = None
    preparer_: Any | None = None
    trainer_: Any | None = None
    label_encoder_: Any | None = None
    history_: dict[str, Any] | None = None

    @property
    def label_mapping_(self) -> dict[str, int] | None:
        if self.label_encoder_ is None:
            return None
        return {
            str(label): int(index)
            for label, index in self.label_encoder_.as_label_mapping().items()
        }

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
            return RunStatus.NOT_AVAILABLE, f'FUTURE fusion classifier is unavailable: {exc}'

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        from fedot_ind.core.models.future import (
            ConfigurableMultimodalFusionClassifier,
            FutureClassifierTrainer,
            FutureTrainingConfig,
        )
        from fedot_ind.core.multimodal.preparation import MultimodalDatasetPreparer

        params = dict(self.params or {})
        preparation_kwargs = dict(params.pop('preparation', {}))
        training_kwargs = dict(params.pop('training', {}))
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
            config=_build_future_preparation_config(preparation_kwargs, modalities)
        )
        train_bundle = self.preparer_.fit_transform(features, target)
        if train_bundle.target is None:
            raise BenchmarkClassificationError('Prepared train bundle is missing targets.')
        if self.preparer_.label_encoder_ is None:
            raise BenchmarkClassificationError(
                'FUTURE adapter requires categorical targets; float targets are not supported.'
            )
        self.label_encoder_ = self.preparer_.label_encoder_
        num_classes = self.label_encoder_.num_classes

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
        validation_fraction = float(training_kwargs.pop('validation_fraction', 0.0))
        drop_last = bool(training_kwargs.pop('drop_last', False))
        allowed_training_keys = {field.name for field in fields(FutureTrainingConfig)}
        unknown_training_keys = sorted(set(training_kwargs) - allowed_training_keys)
        if unknown_training_keys:
            raise BenchmarkClassificationError(
                f'Unsupported FUTURE training params: {unknown_training_keys}'
            )
        training_config = FutureTrainingConfig(**training_kwargs)

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
            batch_size=training_config.batch_size,
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
                batch_size=training_config.batch_size,
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

    def predict(self, features: np.ndarray) -> np.ndarray:
        if (
            self.preparer_ is None
            or self.trainer_ is None
            or self.label_encoder_ is None
        ):
            raise BenchmarkClassificationError(
                'FutureFusionClassifierAdapter must be fitted before prediction.'
            )
        test_bundle = self.preparer_.transform(features)
        predictions = (
            self.trainer_.predict(test_bundle.without_target()).detach().cpu().numpy()
        )
        return self._decode_predicted_labels(predictions)

    def _decode_predicted_labels(self, class_indices: np.ndarray) -> np.ndarray:
        from fedot_ind.core.architecture.preprocessing.label_mapping import (
            LabelMappingError,
        )

        if self.label_encoder_ is None:
            raise BenchmarkClassificationError(
                'FutureFusionClassifierAdapter must be fitted before prediction.'
            )
        try:
            decoded = self.label_encoder_.inverse_transform(
                np.asarray(class_indices).reshape(-1).tolist()
            )
        except LabelMappingError as exc:
            raise BenchmarkClassificationError(str(exc)) from exc
        return np.asarray(decoded, dtype=object)

    def export_artifacts(self) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            'adapter': 'future_fusion_classifier',
            'label_mapping': dict(self.label_mapping_ or {}),
        }
        if self.history_ is not None:
            artifacts['training_history'] = dict(self.history_)
            artifacts['train_duration_s'] = self.history_.get('train_duration_s')
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
    'future_fusion_classifier': FutureFusionClassifierAdapter,
    'future_classifier': FutureFusionClassifierAdapter,
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


def _build_future_preparation_config(
    preparation_kwargs: dict[str, Any],
    modalities: Any,
) -> Any:
    """Build preparation config, deriving modalities from classifier when omitted."""
    from fedot_ind.core.multimodal.configs import (
        build_preparation_config,
        default_transformation_config,
    )
    from fedot_ind.core.multimodal.rules import normalize_unique_modalities

    if preparation_kwargs:
        return build_preparation_config(**preparation_kwargs)

    resolved_modalities = normalize_unique_modalities(
        modalities if modalities is not None else ('raw',)
    )
    defaults = default_transformation_config()
    transformation_config = {
        modality: dict(defaults.get(modality, {}))
        for modality in resolved_modalities
    }
    return build_preparation_config(transformation_config=transformation_config)


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


__all__ = [
    "CLASSIFICATION_ADAPTER_REGISTRY",
    "FutureFusionClassifierAdapter",
    "KernelEnsembleClassifierAdapter",
    "MajorityClassClassifier",
    "NearestCentroidClassifier",
    "OptionalExternalClassifier",
    "PDLClassifierAdapter",
    "build_classification_model",
]

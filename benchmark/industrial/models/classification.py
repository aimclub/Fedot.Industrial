from __future__ import annotations

from dataclasses import dataclass
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
    tags: tuple[str, ...] = ('baseline', 'classification', 'external')
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
    label_mapping_: dict[str, int] | None = None
    inverse_label_mapping_: dict[int, str] | None = None
    history_: dict[str, Any] | None = None

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
        from fedot_ind.core.multimodal.configs import build_preparation_config
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
        training_config = FutureTrainingConfig(
            epochs=int(training_kwargs.pop('epochs', 2)),
            batch_size=int(training_kwargs.pop('batch_size', 32)),
            learning_rate=float(training_kwargs.pop('learning_rate', 1e-3)),
            weight_decay=float(training_kwargs.pop('weight_decay', 0.0)),
            early_stopping_patience=training_kwargs.pop('early_stopping_patience', None),
            device=training_kwargs.pop('device', 'cpu'),
            seed=training_kwargs.pop('seed', 42),
        )
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
            or self.inverse_label_mapping_ is None
        ):
            raise BenchmarkClassificationError(
                'FutureFusionClassifierAdapter must be fitted before prediction.'
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


def build_classification_model(spec: ModelSpec):
    name = spec.adapter_name.lower()
    if name == 'majority_class':
        return MajorityClassClassifier(
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'classification'),
        )
    if name == 'nearest_centroid':
        return NearestCentroidClassifier(
            name=spec.display_name,
            tags=spec.tags or ('baseline', 'classification'),
        )
    if name == 'kernel_ensemble_classifier':
        return KernelEnsembleClassifierAdapter(
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'kernel_learning'),
            optional=spec.optional,
            params=dict(spec.params),
        )
    if name in {'pdl_classifier', 'pdl_clf'}:
        return PDLClassifierAdapter(
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'pdl'),
            optional=True,
            params=dict(spec.params),
        )
    if name in {'future_fusion_classifier', 'future_classifier'}:
        return FutureFusionClassifierAdapter(
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'future', 'multimodal'),
            optional=spec.optional,
            params=dict(spec.params),
        )
    if name == 'fedot_industrial_classifier':
        return OptionalExternalClassifier(
            dependency_name='fedot',
            name=spec.display_name,
            tags=spec.tags or ('industrial', 'classification', 'external'),
        )
    raise BenchmarkClassificationError(
        f'Unsupported classification model adapter: {spec.adapter_name}'
    )


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
    "FutureFusionClassifierAdapter",
    "KernelEnsembleClassifierAdapter",
    "MajorityClassClassifier",
    "NearestCentroidClassifier",
    "OptionalExternalClassifier",
    "PDLClassifierAdapter",
    "build_classification_model",
]

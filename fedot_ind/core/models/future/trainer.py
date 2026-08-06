"""Bundle-aware trainer for FUTURE multimodal classifiers."""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedot_ind.core.models.future.future_clf import ConfigurableMultimodalFusionClassifier
from fedot_ind.core.models.future.tools import (
    FutureTrainingConfig,
    FutureTrainingHistory,
    FusionAuxOutput,
    count_parameters,
)
from fedot_ind.core.multimodal.batching import make_bundle_dataloader
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.operation.transformation.torch_backend.io import resolve_torch_device


class FutureClassifierTrainer:
    """Train/evaluate a :class:`ConfigurableMultimodalFusionClassifier`.

    The trainer owns the training loop only. Callers (or adapters) prepare
    :class:`~torch.utils.data.DataLoader` instances that yield
    :class:`~fedot_ind.core.multimodal.data_bundle.MultimodalDataBundle`
    mini-batches.
    """

    def __init__(
        self,
        model: ConfigurableMultimodalFusionClassifier,
        config: FutureTrainingConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
    ):
        self.model = model
        self.config = config or FutureTrainingConfig()
        self.device = resolve_torch_device(self.config.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.history: FutureTrainingHistory | None = None
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self._build_shapes: dict[MultimodalModality, tuple[int, ...]] | None = None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        build_bundle: MultimodalDataBundle | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
    ) -> FutureTrainingHistory:
        """Fit the wrapped model from externally prepared loaders.

        Parameters
        ----------
        train_loader, val_loader:
            DataLoaders yielding supervised ``MultimodalDataBundle`` batches.
        build_bundle:
            Bundle used to ``build`` the model when encoders/fusion are not
            initialized yet. Required unless ``model.build(...)`` was already
            called.
        optimizer, criterion:
            Optional overrides. Defaults are Adam (lr/weight_decay from config)
            and ``CrossEntropyLoss``. Instances passed to ``__init__`` are used
            when these arguments are omitted.
        """

        self._set_seed(self.config.seed)
        self._ensure_built_for_fit(build_bundle=build_bundle)
        self.model.to(self.device)

        resolved_criterion = self._resolve_criterion(criterion)
        resolved_optimizer = self._resolve_optimizer(optimizer)

        history = FutureTrainingHistory(num_parameters=count_parameters(self.model))
        best_metric = float("inf")
        best_epoch = 0
        patience_counter = 0
        self._best_state_dict = copy.deepcopy(self.model.state_dict())

        started_at = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(
                train_loader,
                criterion=resolved_criterion,
                optimizer=resolved_optimizer,
                train=True,
            )
            history.train_loss.append(train_loss)

            val_loss: float | None = None
            if val_loader is not None:
                val_loss = self._run_epoch(
                    val_loader,
                    criterion=resolved_criterion,
                    optimizer=None,
                    train=False,
                )
            history.validation_loss.append(val_loss)

            monitored = val_loss if val_loss is not None else train_loss
            if monitored < best_metric:
                best_metric = monitored
                best_epoch = epoch
                patience_counter = 0
                self._best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if (
                self.config.early_stopping_patience is not None
                and patience_counter >= self.config.early_stopping_patience
            ):
                history.stopped_early = True
                break

        history.best_epoch = best_epoch
        history.best_validation_loss = (
            None if val_loader is None else best_metric
        )
        history.train_duration_s = time.perf_counter() - started_at

        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

        self.history = history
        return history

    @torch.no_grad()
    def predict_proba(self, bundle: MultimodalDataBundle) -> torch.Tensor:
        """Return class probabilities for each sample in ``bundle``."""

        self._ensure_built(bundle)
        self.model.eval()
        logits = self._predict_logits(bundle)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, bundle: MultimodalDataBundle) -> torch.Tensor:
        """Return predicted class indices for each sample in ``bundle``."""

        probabilities = self.predict_proba(bundle)
        return probabilities.argmax(dim=-1)

    @torch.no_grad()
    def evaluate_diagnostics(self, bundle: MultimodalDataBundle) -> FusionAuxOutput:
        """Run a diagnostic forward pass with fusion aux payload."""

        self._ensure_built(bundle)
        self.model.eval()
        device_bundle = bundle.to(device=self.device)
        aux = self.model(device_bundle, return_aux=True)
        if not isinstance(aux, FusionAuxOutput):
            raise RuntimeError("Expected FusionAuxOutput when return_aux=True.")
        return aux

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist model weights, rebuild metadata, config and history."""

        if self.model.encoders is None or self.model.fusion is None:
            raise ValueError("Model must be built before saving a checkpoint.")
        if self._build_shapes is None:
            raise ValueError("Missing build shapes. Call fit() before save_checkpoint().")

        payload = {
            "model_state_dict": self.model.state_dict(),
            "classifier_config": self._classifier_config(),
            "shapes": {
                modality.value: list(shape)
                for modality, shape in self._build_shapes.items()
            },
            "modalities": [
                modality.value for modality in self._build_shapes
            ],
            "training_config": asdict(self.config),
            "history": None if self.history is None else asdict(self.history),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: Any = "cpu",
        config: FutureTrainingConfig | None = None,
    ) -> "FutureClassifierTrainer":
        """Restore a trainer and rebuilt model from ``save_checkpoint`` output."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        classifier_config = dict(payload["classifier_config"])
        model = ConfigurableMultimodalFusionClassifier(**classifier_config)

        shapes = {
            MultimodalModality(name): tuple(shape)
            for name, shape in payload["shapes"].items()
        }
        modalities = tuple(MultimodalModality(name) for name in payload["modalities"])
        model.build_from_shapes(shapes, bundle_modalities=modalities)

        raw_config = dict(payload["training_config"])
        # Backward-compatible ignore of loader-owned fields from older checkpoints.
        raw_config.pop("validation_fraction", None)
        raw_config.pop("drop_last", None)
        training_config = config or FutureTrainingConfig(**raw_config)
        training_config.device = device
        trainer = cls(model=model, config=training_config)
        trainer._build_shapes = shapes
        trainer.model.load_state_dict(payload["model_state_dict"])
        trainer.model.to(trainer.device)

        history_payload = payload.get("history")
        if history_payload is not None:
            trainer.history = FutureTrainingHistory(**history_payload)
        trainer._best_state_dict = copy.deepcopy(trainer.model.state_dict())
        return trainer

    def _resolve_criterion(self, criterion: nn.Module | None) -> nn.Module:
        resolved = criterion if criterion is not None else self.criterion
        if resolved is None:
            resolved = nn.CrossEntropyLoss()
        self.criterion = resolved
        return resolved

    def _resolve_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None,
    ) -> torch.optim.Optimizer:
        resolved = optimizer if optimizer is not None else self.optimizer
        if resolved is None:
            resolved = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        self.optimizer = resolved
        return resolved

    def _run_epoch(
        self,
        loader: Iterable[MultimodalDataBundle],
        *,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        train: bool,
    ) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                self._validate_supervised_bundle(batch, name="batch")
                batch = batch.to(device=self.device)
                logits = self.model(batch.without_target())
                loss = criterion(logits, batch.target)

                if train:
                    if optimizer is None:
                        raise RuntimeError("optimizer is required when train=True.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                batch_size = batch.n_samples
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError("Epoch loader produced zero samples.")
        return total_loss / total_samples

    @torch.no_grad()
    def _predict_logits(self, bundle: MultimodalDataBundle) -> torch.Tensor:
        self.model.eval()
        loader = make_bundle_dataloader(
            bundle,
            batch_size=self.config.batch_size,
            shuffle=False,
            device=self.device,
            seed=self.config.seed,
            drop_last=False,
            require_target=False,
        )
        outputs: list[torch.Tensor] = []
        for batch in loader:
            forward_bundle = batch.without_target() if batch.target is not None else batch
            outputs.append(self.model(forward_bundle))
        return torch.cat(outputs, dim=0)

    def _ensure_built_for_fit(
        self,
        *,
        build_bundle: MultimodalDataBundle | None,
    ) -> None:
        if self.model.encoders is not None and self.model.fusion is not None:
            if self._build_shapes is None and build_bundle is not None:
                self._build_shapes = dict(build_bundle.shapes)
            if self._build_shapes is None:
                raise ValueError(
                    "Model is built but build shapes are unknown. "
                    "Pass build_bundle=... to fit()."
                )
            return

        if build_bundle is None:
            raise ValueError(
                "Model is not built. Call model.build(bundle) first or pass "
                "build_bundle=... to fit()."
            )
        self._validate_supervised_bundle(build_bundle, name="build_bundle")
        self.model.build(build_bundle)
        self._build_shapes = dict(build_bundle.shapes)

    def _ensure_built(self, bundle: MultimodalDataBundle) -> None:
        if self.model.encoders is None or self.model.fusion is None:
            self.model.build(bundle)
            self._build_shapes = dict(bundle.shapes)
            self.model.to(self.device)

    def _classifier_config(self) -> dict[str, Any]:
        modalities = self.model.modalities_config or self.model.modalities
        if modalities is None:
            raise ValueError("Model modalities are not resolved.")
        return {
            "num_classes": self.model.num_classes,
            "fusion_method": self.model.fusion_method.value,
            "d_model": self.model.d_model,
            "modalities": [modality.value for modality in modalities],
            "encoder_kwargs": dict(self.model.encoder_kwargs),
            "fusion_kwargs": dict(self.model.fusion_kwargs),
            "head_hidden_dim": self.model.head_hidden_dim,
            "head_dropout": self.model.head_dropout,
            "head_activation": self.model.head_activation,
            "raw_modality": self.model.raw_modality.value,
        }

    @staticmethod
    def _validate_supervised_bundle(
        bundle: MultimodalDataBundle,
        *,
        name: str,
    ) -> None:
        if bundle.target is None:
            raise ValueError(f"{name} must include a target tensor.")
        if bundle.target.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError(
                f"{name}.target must contain integer class indices, "
                f"got dtype={bundle.target.dtype}."
            )

    @staticmethod
    def _set_seed(seed: int | None) -> None:
        if seed is None:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


__all__ = [
    "FutureClassifierTrainer",
    "FutureTrainingConfig",
    "FutureTrainingHistory",
]

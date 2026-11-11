"""Run a supervised training plan against caller-owned PyTorch objects."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from fedot_ind.neural.training.checkpoint import CpuCheckpoint
from fedot_ind.neural.training.contracts import EpochLoss, TrainingPlan, TrainingProgress, record_epoch


@dataclass(frozen=True)
class SupervisedBatch:
    """One model input (possibly structured) and one batched target tensor."""

    inputs: Any
    target: Tensor


def _tensor_batch(batch: Any) -> SupervisedBatch:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError("Expected a (features, target) batch or an explicit batch_adapter.")
    return SupervisedBatch(*batch)


def _run_epoch(
    model: nn.Module,
    batches: Iterable,
    criterion: Callable[[Tensor, Tensor], Tensor],
    adapt: Callable[[Any], SupervisedBatch],
    optimizer: Optimizer | None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    samples = 0
    with torch.set_grad_enabled(train):
        for raw_batch in batches:
            batch = adapt(raw_batch)
            target = batch.target
            if not isinstance(target, Tensor) or target.ndim == 0 or target.shape[0] == 0:
                raise ValueError("The target must be a non-empty batched tensor.")
            loss = criterion(model(batch.inputs), target)
            if not isinstance(loss, Tensor) or loss.ndim != 0:
                raise ValueError("The loss must be a scalar tensor (mean per sample).")
            value = float(loss.detach().item())
            if not isfinite(value):
                raise ValueError("The batch loss is not finite.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += value * target.shape[0]
            samples += target.shape[0]
    if samples == 0:
        raise RuntimeError("Epoch loader produced zero samples.")
    return total_loss / samples


class TrainingSession:
    """Keep live training weights separate from the best weights for inference.

    The caller owns device placement, seeds, datasets, loss and optimizer.
    Training accepts re-iterable batches. CPU checkpoints require supported
    DataLoaders and a stateless loss; see the package README.
    """

    def __init__(
        self,
        plan: TrainingPlan,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable[[Tensor, Tensor], Tensor],
        train_batches: Iterable,
        validation_batches: Iterable | None = None,
        *,
        batch_adapter: Callable[[Any], SupervisedBatch] = _tensor_batch,
    ) -> None:
        self._plan = plan
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._train = train_batches
        self._validation = validation_batches
        self._adapt = batch_adapter
        self._progress = TrainingProgress()
        self._best = None
        self._failed = False

    @property
    def progress(self) -> TrainingProgress:
        return self._progress

    def run(self, *, until_epoch: int | None = None) -> TrainingProgress:
        """Run up to a total epoch count or stop on exhausted patience."""
        self._ensure_usable()
        end = self._plan.epochs if until_epoch is None else until_epoch
        if type(end) is not int or not self.progress.completed_epochs <= end <= self._plan.epochs:
            raise ValueError("until_epoch must be between completed epochs and the plan's epoch budget.")
        while self.progress.completed_epochs < end and not self.progress.stopped_early:
            # An exception can leave weights updated for only part of the epoch.
            self._failed = True
            train = _run_epoch(self._model, self._train, self._criterion, self._adapt, self._optimizer)
            validation = None if self._validation is None else _run_epoch(
                self._model, self._validation, self._criterion, self._adapt, None)
            progress = record_epoch(self._plan, self.progress, EpochLoss(train, validation))
            if progress.best_epoch != self.progress.best_epoch:
                self._best = deepcopy(self._model.state_dict())
            self._progress = progress
            self._failed = False
        return self.progress

    def best_weights(self) -> dict[str, Tensor]:
        """Return a copy for inference without changing the live training state."""
        self._ensure_usable()
        if self._best is None:
            raise RuntimeError("No epoch has completed.")
        return deepcopy(self._best)

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save the last completed CPU epoch, not just best weights."""
        self._ensure_usable()
        if self._best is None:
            raise RuntimeError("No epoch has completed.")
        self._checkpoint().save(Path(path), self._plan, self.progress, self._best)

    def load_checkpoint(self, path: str | Path) -> TrainingProgress:
        """Restore into a freshly constructed CPU session with the same inputs."""
        self._ensure_usable()
        if self.progress.completed_epochs:
            raise RuntimeError("Restore requires a fresh training session.")
        checkpoint = self._checkpoint()
        snapshot = checkpoint.read(Path(path), self._plan)
        self._failed = True
        checkpoint.restore(snapshot)
        self._progress, self._best = snapshot.progress, snapshot.best_weights
        self._failed = False
        return self.progress

    def _checkpoint(self) -> CpuCheckpoint:
        return CpuCheckpoint(self._model, self._optimizer, self._train, self._validation)

    def _ensure_usable(self) -> None:
        if self._failed:
            raise RuntimeError("Training session failed; restore a checkpoint in a fresh session.")

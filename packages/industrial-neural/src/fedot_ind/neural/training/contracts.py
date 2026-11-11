"""Validated plan and pure epoch transitions for supervised training."""

from dataclasses import dataclass
from math import isfinite


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class TrainingPlan:
    """Total epoch budget, including any epochs completed before resuming."""

    epochs: int
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        _positive_int(self.epochs, "epochs")
        if self.early_stopping_patience is not None:
            _positive_int(self.early_stopping_patience, "early_stopping_patience")


@dataclass(frozen=True)
class EpochLoss:
    train: float
    validation: float | None = None

    def __post_init__(self) -> None:
        losses = (self.train,) if self.validation is None else (self.train, self.validation)
        if any(type(loss) not in (float, int) or not isfinite(loss) for loss in losses):
            raise ValueError("Epoch losses must be finite numbers.")

    @property
    def monitored(self) -> float:
        return self.train if self.validation is None else self.validation


@dataclass(frozen=True)
class TrainingProgress:
    losses: tuple[EpochLoss, ...] = ()
    best_epoch: int | None = None
    stopped_early: bool = False

    @property
    def completed_epochs(self) -> int:
        return len(self.losses)

    @property
    def best_loss(self) -> EpochLoss | None:
        return None if self.best_epoch is None else self.losses[self.best_epoch - 1]


def record_epoch(plan: TrainingPlan, progress: TrainingProgress, loss: EpochLoss) -> TrainingProgress:
    """Keep the first strict minimum, using validation loss when available."""
    if progress.stopped_early or progress.completed_epochs >= plan.epochs:
        raise ValueError("Cannot append to a finished training run.")
    best = progress.best_loss
    if best is not None and (best.validation is None) != (loss.validation is None):
        raise ValueError("The validation loader must stay fixed throughout training.")

    epoch = progress.completed_epochs + 1
    best_epoch = progress.best_epoch
    if best_epoch is None or loss.monitored < progress.losses[best_epoch - 1].monitored:
        best_epoch = epoch
    patience = plan.early_stopping_patience
    stopped = patience is not None and epoch - best_epoch >= patience
    return TrainingProgress(progress.losses + (loss,), best_epoch, stopped)

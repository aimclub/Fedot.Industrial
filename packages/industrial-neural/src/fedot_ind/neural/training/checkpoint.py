"""Versioned epoch-boundary checkpoints for a single CPU training session."""

import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
from torch import Generator, Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from fedot_ind.neural.training.contracts import EpochLoss, TrainingPlan, TrainingProgress, record_epoch

_SCHEMA = "industrial_neural_training@1"


def _loader_setup(train: Iterable, validation: Iterable | None) -> tuple[list[dict], list[Generator]]:
    """Describe supported samplers and preserve sharing of their random streams."""
    generators: list[Generator] = []

    def stream(generator: Generator | None) -> int | None:
        if generator is None:
            return None
        if generator.device.type != "cpu":
            raise ValueError("Checkpoint loader generators must be on CPU.")
        for index, existing in enumerate(generators):
            if generator is existing:
                return index
        generators.append(generator)
        return len(generators) - 1

    signatures = []
    for loader in (train,) if validation is None else (train, validation):
        if (type(loader) is not DataLoader or loader.num_workers != 0
                or type(loader.batch_sampler) is not BatchSampler or loader.batch_size is None):
            raise ValueError("CPU checkpoints require DataLoader with automatic batching and num_workers=0.")
        sampler = loader.sampler
        if type(sampler) not in (RandomSampler, SequentialSampler):
            raise ValueError("CPU checkpoints require a standard sequential or random sampler.")
        loader_stream = stream(loader.generator)
        signatures.append(dict(
            sampler=type(sampler).__name__,
            samples=len(sampler),
            dataset_size=len(loader.dataset),
            batch_size=loader.batch_size,
            drop_last=loader.drop_last,
            replacement=type(sampler) is RandomSampler and sampler.replacement,
            loader_stream=loader_stream,
            sampler_stream=stream(sampler.generator) if type(sampler) is RandomSampler else None,
        ))
    return signatures, generators


def _optimizer_layout(model: nn.Module, optimizer: Optimizer) -> dict:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        groups = [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]
    except KeyError as exc:
        raise ValueError("Checkpoint optimizer parameters must belong to the model.") from exc
    cls = type(optimizer)
    return dict(kind=f"{cls.__module__}.{cls.__qualname__}", groups=groups)


def _check_weights(weights: Any, expected: dict[str, Tensor], label: str) -> None:
    if not isinstance(weights, dict) or weights.keys() != expected.keys():
        raise ValueError(f"Checkpoint {label} keys do not match the model.")
    for name, tensor in weights.items():
        current = expected[name]
        if (not isinstance(tensor, Tensor) or not isinstance(current, Tensor)
                or tensor.shape != current.shape or tensor.dtype != current.dtype):
            raise ValueError(f"Checkpoint {label} tensor does not match the model: {name}.")


@dataclass(frozen=True)
class _RandomState:
    python: tuple
    numpy: tuple
    torch_cpu: Tensor
    loaders: tuple[Tensor, ...]

    @classmethod
    def capture(cls, generators: list[Generator]) -> dict:
        state = np.random.get_state()
        if not isinstance(state, tuple):
            raise ValueError("CPU checkpoints require NumPy's global MT19937 random stream.")
        kind, keys, pos, has_gauss, cached = state
        return dict(
            python=random.getstate(),
            numpy=(kind, torch.from_numpy(keys.astype("int64")), pos, has_gauss, cached),
            torch_cpu=torch.get_rng_state(),
            loaders=[generator.get_state() for generator in generators],
        )

    @classmethod
    def parse(cls, raw: dict, generator_count: int) -> "_RandomState":
        try:
            kind, keys, pos, has_gauss, cached = raw["numpy"]
            if (kind != "MT19937" or keys.dtype != torch.int64 or keys.shape != (624,)
                    or bool(((keys < 0) | (keys >= 2**32)).any()) or not 0 <= pos <= 624):
                raise ValueError("Invalid NumPy random state.")
            numpy_state = (kind, keys.numpy().astype("uint32"), pos, has_gauss, cached)
            loaders = tuple(raw["loaders"])
            if len(loaders) != generator_count:
                raise ValueError("Checkpoint loader generators do not match.")
            # Validate against private generators before changing any global state.
            random.Random().setstate(raw["python"])
            np.random.RandomState().set_state(numpy_state)
            for state in (raw["torch_cpu"], *loaders):
                Generator().set_state(state)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError("Invalid checkpoint random state.") from exc
        return cls(raw["python"], numpy_state, raw["torch_cpu"], loaders)

    def restore(self, generators: list[Generator]) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch_cpu)
        for generator, state in zip(generators, self.loaders, strict=True):
            generator.set_state(state)


@dataclass(frozen=True)
class _Snapshot:
    progress: TrainingProgress
    model_weights: dict[str, Tensor]
    best_weights: dict[str, Tensor]
    optimizer_state: dict
    random_state: _RandomState


class CpuCheckpoint:
    def __init__(self, model: nn.Module, optimizer: Optimizer, train: Iterable, validation: Iterable | None):
        if any(tensor.device.type != "cpu" for tensor in chain(model.parameters(), model.buffers())):
            raise ValueError("Training checkpoints currently support CPU models only.")
        self._model = model
        self._optimizer = optimizer
        self._loaders, self._generators = _loader_setup(train, validation)
        self._layout = _optimizer_layout(model, optimizer)

    def save(
        self, path: Path, plan: TrainingPlan, progress: TrainingProgress, best_weights: dict[str, Tensor]
    ) -> None:
        weights = self._model.state_dict()
        _check_weights(best_weights, weights, "best")
        payload = dict(
            schema=_SCHEMA,
            plan=asdict(plan),
            losses=[asdict(loss) for loss in progress.losses],
            model=weights,
            best=best_weights,
            optimizer_layout=self._layout,
            optimizer=self._optimizer.state_dict(),
            loaders=self._loaders,
            rng=_RandomState.capture(self._generators),
        )
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def read(self, path: Path, plan: TrainingPlan) -> _Snapshot:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
            raise ValueError("Unsupported training checkpoint schema.")
        if payload.get("plan") != asdict(plan):
            raise ValueError("Checkpoint training plan does not match.")
        if payload.get("loaders") != self._loaders:
            raise ValueError("Checkpoint DataLoader configuration does not match.")
        if payload.get("optimizer_layout") != self._layout:
            raise ValueError("Checkpoint optimizer type or parameter groups do not match.")

        progress = TrainingProgress()
        raw_losses = payload.get("losses")
        if not isinstance(raw_losses, list) or not 0 < len(raw_losses) <= plan.epochs:
            raise ValueError("Checkpoint must contain a valid completed epoch history.")
        for item in raw_losses:
            if not isinstance(item, dict) or item.keys() != {"train", "validation"}:
                raise ValueError("Invalid checkpoint epoch loss.")
            loss = EpochLoss(**item)
            if (loss.validation is not None) != (len(self._loaders) == 2):
                raise ValueError("Checkpoint validation history does not match the loaders.")
            progress = record_epoch(plan, progress, loss)

        expected = self._model.state_dict()
        _check_weights(payload.get("model"), expected, "model")
        _check_weights(payload.get("best"), expected, "best")
        rng = _RandomState.parse(payload.get("rng"), len(self._generators))
        optimizer = payload.get("optimizer")
        if not isinstance(optimizer, dict) or optimizer.keys() != {"state", "param_groups"}:
            raise ValueError("Invalid checkpoint optimizer state.")
        return _Snapshot(progress, payload["model"], payload["best"], optimizer, rng)

    def restore(self, snapshot: _Snapshot) -> None:
        self._model.load_state_dict(snapshot.model_weights)
        self._optimizer.load_state_dict(snapshot.optimizer_state)
        snapshot.random_state.restore(self._generators)

"""Check CPU restart equivalence on two small, offline time-series tasks."""

import argparse
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedot_ind.neural.blocks.linear import LinLnDrop
from fedot_ind.neural.encoders import build_encoder, raw_encoder_config
from fedot_ind.neural.training import TrainingPlan, TrainingSession


_TASKS = ("classification", "forecasting")
_PLAN = TrainingPlan(epochs=4, early_stopping_patience=3)


def _classification():
    signals = torch.randn(26, 1, 24, generator=torch.Generator().manual_seed(13))
    labels = (signals.mean(dim=(1, 2)) > 0).long()
    encoder = build_encoder(raw_encoder_config(in_channels=1, d_model=8))
    model = nn.Sequential(encoder, LinLnDrop(8, 2, p=0.2))
    return model, nn.CrossEntropyLoss(), TensorDataset(signals[:19], labels[:19]), TensorDataset(
        signals[19:], labels[19:])


def _forecasting():
    lookback, horizon = 8, 3
    time = torch.arange(75, dtype=torch.float32)
    series = torch.sin(time / 5) + time / 100

    def windows(segment):
        values = segment.unfold(0, lookback + horizon, 1)
        return TensorDataset(values[:, :lookback], values[:, lookback:])

    # Chronological split before windowing: no shared observations between splits.
    return LinLnDrop(lookback, horizon, p=0.2), nn.MSELoss(), windows(series[:50]), windows(series[50:])


def _train(task, output, *, resume=None, until_epoch=None):
    seed = 19 if resume is None else 97
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, loss, train, validation = _classification() if task == "classification" else _forecasting()
    train_loader = DataLoader(train, batch_size=6, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(validation, batch_size=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    session = TrainingSession(_PLAN, model, optimizer, loss, train_loader, val_loader)
    if resume is not None:
        session.load_checkpoint(resume)
    progress = session.run(until_epoch=until_epoch)
    session.save_checkpoint(output)
    return progress


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        for a, b in zip(actual, expected, strict=True):
            _assert_equal(a, b)
    else:
        assert actual == expected


def _verify(task, directory):
    reference, interrupted, resumed = (directory / f"{task}-{phase}.pt"
                                       for phase in ("reference", "interrupted", "resumed"))
    progress = _train(task, reference)
    prefix = _train(task, interrupted, until_epoch=2)
    assert prefix.losses == progress.losses[:2]

    isolated = ["-I"] if sys.flags.isolated else []
    subprocess.run(
        [sys.executable, *isolated, str(Path(__file__).resolve()), "--task", task,
         "--resume", str(interrupted), "--save", str(resumed)],
        check=True, timeout=60,
    )
    _assert_equal(
        torch.load(resumed, weights_only=True, map_location="cpu"),
        torch.load(reference, weights_only=True, map_location="cpu"),
    )
    print(f"{task}: {progress.completed_epochs} epochs, best={progress.best_epoch}; exact CPU restart")


def _require_standalone():
    for module in ("fedot", "golem", "fedot_ind.core"):
        if find_spec(module) is not None:
            raise RuntimeError(f"Expected a clean neural environment; found {module}.")
    try:
        metadata.version("fedot-ind")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("fedot-ind is installed in the standalone environment.")
    dist = metadata.distribution("industrial-neural")
    origin = Path(find_spec("fedot_ind.neural").origin).resolve()
    expected = Path(dist.locate_file("fedot_ind/neural/__init__.py")).resolve()
    if origin != expected or not origin.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Neural package did not load from the installed wheel: {origin}.")
    print(f"industrial-neural {dist.version}, Python {sys.version.split()[0]}, torch {torch.__version__}; no FEDOT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", action="store_true", help="Require only the installed neural distribution")
    parser.add_argument("--task", choices=_TASKS)
    parser.add_argument("--save", type=Path, help="Save one run instead of comparing runs")
    parser.add_argument("--resume", type=Path, help="Restore this CPU checkpoint before training")
    parser.add_argument("--until-epoch", type=int, help="Pause after this total epoch count")
    args = parser.parse_args()
    if args.save is not None and args.task is None:
        parser.error("--save requires --task")
    if args.save is None and (args.resume is not None or args.until_epoch is not None):
        parser.error("--resume and --until-epoch require --save")
    if args.standalone:
        _require_standalone()
    torch.set_num_threads(1)
    if args.save is not None:
        _train(args.task, args.save, resume=args.resume, until_epoch=args.until_epoch)
    else:
        with TemporaryDirectory() as temp:
            for task in _TASKS if args.task is None else (args.task,):
                _verify(task, Path(temp))


if __name__ == "__main__":
    main()

import random

from hypothesis import given, strategies as st
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from fedot_ind.neural.training import EpochLoss, TrainingPlan, TrainingProgress, TrainingSession
from fedot_ind.neural.training.contracts import record_epoch


@given(st.lists(st.tuples(st.integers(-100, 100), st.integers(-100, 100)), min_size=1, max_size=25))
def test_progress_uses_first_validation_minimum_and_keeps_order(pairs):
    plan = TrainingPlan(epochs=len(pairs))
    progress = TrainingProgress()
    losses = tuple(EpochLoss(*pair) for pair in pairs)
    for loss in losses:
        progress = record_epoch(plan, progress, loss)
    assert progress.losses == losses
    assert progress.best_epoch == min(range(len(pairs)), key=lambda index: pairs[index][1]) + 1
    assert not progress.stopped_early
    with pytest.raises(ValueError, match="finished"):
        record_epoch(plan, progress, losses[0])


@pytest.mark.parametrize("epochs, patience", [(0, None), (True, None), (3, 0), (3, 1.5)])
def test_plan_rejects_invalid_epoch_budget(epochs, patience):
    with pytest.raises(ValueError, match="positive integer"):
        TrainingPlan(epochs, patience)


def test_loss_contract_and_validation_cannot_change_between_epochs():
    with pytest.raises(ValueError, match="finite"):
        EpochLoss(float("nan"))
    plan = TrainingPlan(2)
    progress = record_epoch(plan, TrainingProgress(), EpochLoss(2))
    with pytest.raises(ValueError, match="validation"):
        record_epoch(plan, progress, EpochLoss(1, 0))


def test_epoch_loss_is_sample_weighted_and_validation_disables_gradients():
    modes = []

    class Model(nn.Linear):
        def forward(self, x):
            modes.append((self.training, torch.is_grad_enabled()))
            return super().forward(x)

    model = Model(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    batches = [(torch.ones(3, 1), torch.ones(3, 1)), (torch.ones(1, 1), torch.full((1, 1), 3.))]
    session = TrainingSession(TrainingPlan(1), model, optimizer, nn.MSELoss(), batches, batches)
    assert session.run().losses == (EpochLoss(3., 3.),)
    assert modes == [(True, True)] * 2 + [(False, False)] * 2


@pytest.mark.parametrize("batch, loss, error, message", [
    ([], nn.MSELoss(), RuntimeError, "zero samples"),
    ([(torch.ones(1, 1), torch.ones(()))], nn.MSELoss(), ValueError, "batched tensor"),
    ([(torch.ones(1, 1), torch.ones(1, 1))], nn.MSELoss(reduction="none"), ValueError, "scalar tensor"),
    ([(torch.ones(1, 1), torch.full((1, 1), float("inf")))], nn.MSELoss(), ValueError, "not finite"),
    (["missing target"], nn.MSELoss(), TypeError, "batch_adapter"),
])
def test_failed_epoch_cannot_be_resumed_or_saved(batch, loss, error, message, tmp_path):
    model = nn.Linear(1, 1)
    session = TrainingSession(TrainingPlan(2), model, torch.optim.SGD(model.parameters(), lr=0.1), loss, batch)
    with pytest.raises(error, match=message):
        session.run()
    assert session.progress.completed_epochs == 0
    with pytest.raises(RuntimeError, match="session failed"):
        session.run()
    with pytest.raises(RuntimeError, match="session failed"):
        session.save_checkpoint(tmp_path / "failed.pt")
    assert not list(tmp_path.iterdir())


def _session(*, plan=TrainingPlan(5, 2), batch_size=4, reverse_params=False, device="cpu",
             num_workers=0, custom_sampler=False, validation=True):
    model = nn.Linear(1, 1, device=device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(parameters[::-1] if reverse_params else parameters, lr=0.1, momentum=0.1)
    inputs = torch.ones(4, 1)
    train = TensorDataset(inputs, inputs)
    loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers,
                        sampler=SubsetRandomSampler(range(len(train))) if custom_sampler else None)
    val_loader = DataLoader(TensorDataset(inputs, torch.zeros_like(inputs)), batch_size=batch_size)
    session = TrainingSession(plan, model, optimizer, nn.MSELoss(), loader, val_loader if validation else None)
    return session, model, optimizer


def test_checkpoint_keeps_last_weights_rng_and_patience(tmp_path):
    session, model, optimizer = _session()
    path = tmp_path / "resume.pt"
    with pytest.raises(RuntimeError, match="No epoch"):
        session.save_checkpoint(path)
    progress = session.run(until_epoch=2)
    assert progress.best_epoch == 1
    assert not progress.stopped_early
    best = session.best_weights()
    assert not torch.equal(model.weight, best["weight"])
    best["weight"].zero_()
    assert torch.count_nonzero(session.best_weights()["weight"])
    with pytest.raises(ValueError, match="until_epoch"):
        session.run(until_epoch=1)
    session.save_checkpoint(path)
    expected_random = (random.random(), np.random.rand(), torch.rand(1).item())

    restored, replacement, replacement_optimizer = _session()
    assert restored.load_checkpoint(path) == progress
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected_random
    torch.testing.assert_close(replacement.state_dict(), model.state_dict(), rtol=0, atol=0)
    assert len(replacement_optimizer.state) == len(optimizer.state) > 0
    for actual, expected in zip(replacement_optimizer.state.values(), optimizer.state.values(), strict=True):
        torch.testing.assert_close(actual["momentum_buffer"], expected["momentum_buffer"], rtol=0, atol=0)
    torch.testing.assert_close(restored.best_weights(), session.best_weights(), rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="fresh training session"):
        restored.load_checkpoint(path)
    after = restored.run()
    assert after.stopped_early
    assert after.completed_epochs == 3
    assert after.best_epoch == 1
    assert restored.run() == after


@pytest.mark.parametrize("options, message", [
    ({"plan": TrainingPlan(6)}, "plan does not match"),
    ({"batch_size": 3}, "DataLoader configuration"),
    ({"reverse_params": True}, "parameter groups"),
    ({"validation": False}, "DataLoader configuration"),
    ({"num_workers": 1}, "num_workers=0"),
    ({"custom_sampler": True}, "standard sequential or random sampler"),
    ({"device": "meta"}, "CPU models only"),
])
def test_restore_rejects_incompatible_sessions(options, message, tmp_path):
    path = tmp_path / "resume.pt"
    source, _, _ = _session()
    source.run(until_epoch=1)
    source.save_checkpoint(path)
    incompatible, _, _ = _session(**options)
    with pytest.raises(ValueError, match=message):
        incompatible.load_checkpoint(path)
    assert incompatible.progress.completed_epochs == 0


@pytest.mark.parametrize("field, value, message", [
    ("schema", "obsolete", "schema"),
    ("losses", [], "epoch history"),
    ("model", {"weight": torch.ones(1, 1)}, "model keys"),
    ("best", {"weight": torch.ones(2, 1), "bias": torch.zeros(1)}, "best tensor"),
    ("rng", {}, "random state"),
    ("optimizer", {"state": {}}, "optimizer state"),
])
def test_restore_rejects_invalid_payload_before_mutating_state(field, value, message, tmp_path):
    path = tmp_path / "resume.pt"
    source, _, _ = _session()
    source.run(until_epoch=1)
    source.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    restored, model, optimizer = _session()
    with pytest.raises(ValueError, match=message):
        restored.load_checkpoint(path)
    assert restored.progress.completed_epochs == 0
    assert not torch.count_nonzero(model.weight)
    assert optimizer.state == {}


def test_failed_checkpoint_write_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "resume.pt"
    session, _, _ = _session()
    session.run(until_epoch=1)
    session.save_checkpoint(path)
    saved = path.read_bytes()

    def fail(_, stream):
        stream.write(b"partial file")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="simulated disk failure"):
        session.save_checkpoint(path)
    assert path.read_bytes() == saved
    assert list(tmp_path.iterdir()) == [path]

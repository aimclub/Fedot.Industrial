import copy

import torch

from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot_ind.core.models.nn.network_modules.layers.forecasting.nbeats import NBeatsNet


def test_neural_model_cache_restores_tensor_weights_and_removes_checkpoint(tmp_path, monkeypatch):
    class CacheModel(BaseNeuralModel):
        def __repr__(self):
            return "checkpoint_regression"

    monkeypatch.chdir(tmp_path)
    operation = CacheModel({"epochs": 1, "batch_size": 1})
    operation.model = torch.nn.Linear(2, 1)
    expected = copy.deepcopy(operation.model.state_dict())
    operation.model_for_inference = torch.nn.Linear(2, 1)

    operation._save_and_clear_cache()

    assert operation.model is operation.model_for_inference
    assert all(torch.equal(operation.model.state_dict()[name], value) for name, value in expected.items())
    assert next(operation.model.parameters()).device.type == "cpu"
    assert list(tmp_path.glob("*.pth")) == []


def test_nbeats_full_model_checkpoint_keeps_explicit_pickle_loading_compatible(tmp_path):
    model = NBeatsNet(
        device=torch.device("cpu"), stack_types=(NBeatsNet.GENERIC_BLOCK,),
        nb_blocks_per_stack=1, thetas_dim=(2,), hidden_layer_units=8,
    )
    inputs = torch.randn(2, model.backcast_length)
    checkpoint = tmp_path / "nbeats.pt"
    model.save(checkpoint)

    restored = NBeatsNet.load(checkpoint, map_location="cpu")

    assert isinstance(restored, NBeatsNet)
    for actual, expected in zip(restored(inputs), model(inputs)):
        torch.testing.assert_close(actual, expected)

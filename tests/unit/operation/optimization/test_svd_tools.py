import copy
import os

import pytest
from torchvision.models import resnet18

from core.architecture.utils.utils import PROJECT_PATH
from core.operation.optimization.svd_tools import *


def test_decomposition_of_layer():
    energy_threshold_pruning = create_energy_svd_pruning(energy_threshold=0.5)
    conv = DecomposedConv2d(3, 6, 3, decomposing=False)
    with pytest.raises(AssertionError):
        energy_threshold_pruning(conv=conv)


def test_energy_threshold_range():
    with pytest.raises(AssertionError):
        e = create_energy_svd_pruning(energy_threshold=2)
    with pytest.raises(AssertionError):
        e = create_energy_svd_pruning(energy_threshold=0)


def test_energy_threshold_pruning():
    conv = DecomposedConv2d(3, 6, 3)
    conv.set_U_S_Vh(
        u=torch.rand((6, 6)),
        s=torch.Tensor([6, 1, 4, 5, 3, 2]),
        vh=torch.rand(6, 27)
    )
    conv1 = copy.deepcopy(conv)
    energy_threshold_pruning = create_energy_svd_pruning(energy_threshold=1)
    energy_threshold_pruning(conv=conv1)
    assert torch.equal(conv1.S, torch.Tensor([1, 2, 3, 4, 5, 6]))
    assert torch.equal(conv1.U.data, conv.U.data[:, [1, 5, 4, 2, 3, 0]])
    assert torch.equal(conv1.Vh.data, conv.Vh.data[[1, 5, 4, 2, 3, 0], :])
    conv05 = copy.deepcopy(conv)
    energy_threshold_pruning = create_energy_svd_pruning(energy_threshold=0.5)
    energy_threshold_pruning(conv=conv05)
    assert torch.equal(conv05.S, torch.Tensor([5, 6]))
    assert torch.equal(conv05.U.data, conv.U.data[:, [3, 0]])
    assert torch.equal(conv05.Vh.data, conv.Vh.data[[3, 0], :])
    conv01 = copy.deepcopy(conv)
    energy_threshold_pruning = create_energy_svd_pruning(energy_threshold=0.1)
    energy_threshold_pruning(conv=conv01)
    assert torch.equal(conv01.S, torch.Tensor([6]))
    assert torch.equal(conv01.U.data, conv.U.data[:, [0]])
    assert torch.equal(conv01.Vh.data, conv.Vh.data[[0], :])


def test_decompose_module():
    model0 = resnet18()
    model = copy.deepcopy(model0)
    decompose_module(model=model)
    sd0 = model0.state_dict()
    sd = model.state_dict()
    assert sd0.keys() == sd.keys()
    for k in sd0.keys():
        assert torch.equal(sd[k], sd0[k])


def test_load_svd_channel_state_dict():
    svd_state_dict_path = os.path.join(PROJECT_PATH, 'tests/data/cv_test_models/ResNet18_svd_channel.sd.pt')
    svd_model = resnet18(num_classes=3)
    load_svd_state_dict(
        svd_model,
        decomposing_mode='channel',
        state_dict_path=svd_state_dict_path
    )
    svd_state_dict = torch.load(svd_state_dict_path)
    assert svd_state_dict.keys() == svd_model.state_dict().keys()
    for k in svd_state_dict:
        assert torch.equal(svd_model.state_dict()[k], svd_state_dict[k])


def test_load_svd_spatial_state_dict():
    svd_state_dict_path = os.path.join(PROJECT_PATH, 'tests/data/cv_test_models/ResNet18_svd_spatial.sd.pt')
    svd_model = resnet18(num_classes=3)
    load_svd_state_dict(
        svd_model,
        decomposing_mode='spatial',
        state_dict_path=svd_state_dict_path
    )
    svd_state_dict = torch.load(svd_state_dict_path)
    assert svd_state_dict.keys() == svd_model.state_dict().keys()
    for k in svd_state_dict:
        assert torch.equal(svd_model.state_dict()[k], svd_state_dict[k])

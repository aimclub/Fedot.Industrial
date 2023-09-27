import os
from collections import OrderedDict

import pytest
import torch
from torchvision.models import resnet18

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.operation.optimization.sfp_tools import _check_nonzero_filters, \
    _prune_filters, _index_union, _indexes_of_tensor_values, _parse_sd, _collect_sd, \
    create_percentage_filter_zeroing_fn, load_sfp_resnet_model


def test_percentage_filter_zeroing():
    conv = torch.nn.Conv2d(3, 4, 3)
    with pytest.raises(AssertionError):
        create_percentage_filter_zeroing_fn(pruning_ratio=1)
    with pytest.raises(AssertionError):
        create_percentage_filter_zeroing_fn(pruning_ratio=0)
    with torch.no_grad():
        for i, v in enumerate([0.1, 0.5, 0.7, 0.2]):
            conv.weight[i, :, :, :] = v
    percentage_filter_zeroing = create_percentage_filter_zeroing_fn(pruning_ratio=0.5)
    percentage_filter_zeroing(conv=conv)
    expected = torch.zeros_like(conv.weight)
    expected[1, :, :, :] = 0.5
    expected[2, :, :, :] = 0.7
    assert torch.equal(conv.weight, expected)


def test_check_nonzero_filters():
    weight = torch.rand((4, 3, 3, 3))
    weight[2, :, :, :] = 0
    assert torch.equal(_check_nonzero_filters(weight), torch.tensor([0, 1, 3]))
    weight = torch.rand((6, 3, 3, 3))
    weight[[1, 3, 5], :, :, :] = 0
    assert torch.equal(_check_nonzero_filters(weight), torch.tensor([0, 2, 4]))


def test_prune_filters():
    weight = torch.rand((6, 3, 3, 3))
    result = _prune_filters(weight, saving_filters=torch.tensor([2, 4]))
    assert torch.equal(result, weight[[2, 4], :, :, :])
    result = _prune_filters(weight, saving_channels=torch.tensor([2]))
    assert torch.equal(result, weight[:, [2], :, :])


def test_index_union():
    x = torch.tensor([1, 3, 5, 2])
    y = torch.tensor([1, 4, 6, 2])
    assert torch.equal(_index_union(x, y), torch.tensor([1, 2, 3, 4, 5, 6]))


def test_indexes_of_tensor_values():
    x = torch.tensor([1, 3, 5, 2])
    y = torch.tensor([1, 2])
    assert torch.equal(_indexes_of_tensor_values(x, y), torch.tensor([0, 3]))


def test_parse_collect_sd():
    sd = OrderedDict([
        ('conv1.weight', torch.tensor([[1, 2], [5, 4]])),
        ('layer1.bn2.something', torch.tensor([7, 8, 9]))
    ])
    parsed_sd = _parse_sd(sd)
    assert torch.equal(parsed_sd['conv1']['weight'], torch.tensor([[1, 2], [5, 4]]))
    assert torch.equal(parsed_sd['layer1']['bn2']['something'], torch.tensor([7, 8, 9]))
    sd2 = _collect_sd(parsed_sd)
    assert sd.keys() == sd2.keys()
    for k in sd.keys():
        assert torch.equal(sd[k], sd2[k])


def test_load_sfp_resnet_model():
    sfp_state_dict_path = os.path.join(PROJECT_PATH, 'tests/data/cv_test_models/ResNet18_sfp.sd.pt')
    sfp_model = load_sfp_resnet_model(
        model=resnet18(num_classes=3),
        state_dict_path=sfp_state_dict_path,
    )

    sfp_state_dict = torch.load(sfp_state_dict_path, map_location='cpu')
    assert sfp_state_dict.keys() == sfp_model.state_dict().keys()
    for k in sfp_state_dict:
        assert torch.equal(sfp_model.state_dict()[k], sfp_state_dict[k])

from collections import OrderedDict
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.linalg import vector_norm
from torch.nn import Conv2d


def zerolize_filters(conv: Conv2d, pruning_ratio: float) -> None:
    """Zerolize filters of convolutional layer to the pruning_ratio (in-place).

    Args:
        conv: The optimizable layer.
        pruning_ratio: pruning hyperparameter, percentage of zerolized filters.
    """
    filter_pruned_num = int(conv.weight.size()[0] * pruning_ratio)
    filter_norms = vector_norm(conv.weight, dim=(1, 2, 3))
    _, indices = filter_norms.sort()
    with torch.no_grad():
        conv.weight[indices[:filter_pruned_num]] = 0


def _check_zero_filters(weight: Tensor) -> Tensor:
    """Returns indices of zero filters."""
    filters = torch.count_nonzero(weight, dim=(1, 2, 3))
    indices = torch.flatten(torch.nonzero(filters))
    return indices


def _prune_filters(
        weight: Tensor,
        saving_filters: Optional[Tensor] = None,
        saving_channels: Optional[Tensor] = None,
) -> Tensor:
    """Prune filters and channels of convolutional layer.

    Args:
        weight: Weight matrix.
        saving_filters: Indexes of filters to be saved.
            If ``None`` all filters to be saved.
        saving_channels: Indexes of channels to be saved.
            If ``None`` all channels to be saved.
    """
    if saving_filters is not None:
        weight = weight[saving_filters].clone()
    if saving_channels is not None:
        weight = weight[:, saving_channels].clone()
    return weight


def _prune_batchnorm(bn: Dict, saving_channels: Tensor) -> Dict[str, Tensor]:
    """Prune BatchNorm2d.

    Args:
        bn: Dictionary with batchnorm params.
        saving_channels: Indexes of channels to be saved.
            If ``None`` all channels to be saved.
    """
    bn['weight'] = bn['weight'][saving_channels].clone()
    bn['bias'] = bn['bias'][saving_channels].clone()
    bn['running_mean'] = bn['running_mean'][saving_channels].clone()
    bn['running_var'] = bn['running_var'][saving_channels].clone()
    return bn


def _index_union(x: Tensor, y: Tensor) -> Tensor:
    """Returns the union of x and y"""
    x = set(x.tolist())
    y = set(y.tolist())
    xy = x | y
    return torch.tensor(list(xy))


def _parse_resnet_sd(state_dict: OrderedDict):
    """Parses state_dict to nested dictionaries."""
    parsed_sd = OrderedDict()
    for k, v in state_dict.items():
        _parse_resnet_param(k.split('.'), v, parsed_sd)
    return parsed_sd


def _parse_resnet_param(param, value, dictionary):
    """Parses value from state_dict to nested dictionaries."""
    if len(param) > 1:
        dictionary.setdefault(param[0], OrderedDict())
        _parse_resnet_param(param[1:], value, dictionary[param[0]])
    else:
        dictionary[param[0]] = value


def _collect_resnet_sd(keys, parsed_state_dict):
    """Collect state_dict from nested dictionaries."""
    state_dict = OrderedDict()
    for key in keys:
        state_dict[key] = _collect_resnet_param(key.split('.'), parsed_state_dict)
    return state_dict


def _collect_resnet_param(param, dictionary):
    """Gets value from nested dictionaries."""
    if len(param) > 1:
        return _collect_resnet_param(param[1:], dictionary[param[0]])
    else:
        return dictionary[param[0]]


def _prune_resnet_block(block: Dict):
    """Prune block of ResNet"""
    keys = list(block.keys())
    if 'downsample' in keys:
        keys.remove('downsample')
    final_key = keys[-2]
    keys = keys[:-2]
    for key in keys:
        if key.startswith('conv'):
            filters = _check_zero_filters(block[key]['weight'])
            block[key]['weight'] = _prune_filters(weight=block[key]['weight'], saving_filters=filters)
            channels = filters
        elif key.startswith('bn'):
            block[key] = _prune_batchnorm(bn=block[key], saving_channels=channels)
    block[final_key]['weight'] = _prune_filters(weight=block[final_key]['weight'], saving_channels=channels)


def prune_resnet_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Prune state_dict of ResNet

    Args:
        state_dict: ``state_dict`` of ResNet model.

    Returns:
        Pruned ``state_dict``.
    """
    sd = _parse_resnet_sd(state_dict)
    for k, v in sd['layer1'].items():
        _prune_resnet_block(v)
    for k, v in sd['layer2'].items():
        _prune_resnet_block(v)
    for k, v in sd['layer3'].items():
        _prune_resnet_block(v)
    for k, v in sd['layer4'].items():
        _prune_resnet_block(v)
    sd = _collect_resnet_sd(state_dict.keys(), sd)
    return sd

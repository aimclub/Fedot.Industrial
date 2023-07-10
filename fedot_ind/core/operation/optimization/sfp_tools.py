from collections import OrderedDict
from typing import Dict, Optional, Tuple, List, Union, Callable

import torch
from torch import Tensor
from torch.linalg import vector_norm
from torch.nn import Conv2d
from torchvision.models import ResNet

from fedot_ind.core.models.cnn.pruned_resnet import PRUNED_MODELS, PrunedResNet

MODELS_FROM_LENGHT = {
    122: 'ResNet18',
    218: 'ResNet34',
    320: 'ResNet50',
    626: 'ResNet101',
    932: 'ResNet152',
}


def percentage_filter_zeroing(conv: Conv2d, pruning_ratio: float) -> None:
    """Zero filters of convolutional layer to the pruning_ratio (in-place).

    Args:
        conv: The optimizable layer.
        pruning_ratio: pruning hyperparameter must be in the range (0, 1),
            percentage of zeroed filters.
    Raises:
        Assertion Error: If ``energy_threshold`` is not in (0, 1).
    """
    assert 0 < pruning_ratio < 1, "pruning_ratio must be in the range (0, 1)"
    filter_pruned_num = int(conv.weight.size()[0] * pruning_ratio)
    filter_norms = vector_norm(conv.weight, dim=(1, 2, 3))
    _, indices = filter_norms.sort()
    with torch.no_grad():
        conv.weight[indices[:filter_pruned_num]] = 0


def energy_filter_zeroing(conv: Conv2d, energy_threshold: float) -> None:
    """Zero filters of convolutional layer to the energy_threshold (in-place).

    Args:
        conv: The optimizable layer.
        energy_threshold: pruning hyperparameter must be in the range (0, 1].
            the lower the threshold, the more filters will be pruned.
    Raises:
        Assertion Error: If ``energy_threshold`` is not in (0, 1].
    """
    assert 0 < energy_threshold <= 1, "energy_threshold must be in the range (0, 1]"
    filter_norms = vector_norm(conv.weight, dim=(1, 2, 3))
    sorted_filter_norms, indices = filter_norms.sort()
    sum = (filter_norms ** 2).sum()
    threshold = energy_threshold * sum
    for index, filter_norm in zip(indices, sorted_filter_norms):
        with torch.no_grad():
            conv.weight[index] = 0
        sum -= filter_norm ** 2
        if sum < threshold:
            break


def _check_nonzero_filters(weight: Tensor) -> Tensor:
    """Returns indices of nonzero filters."""
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


def _indexes_of_tensor_values(tensor: Tensor, values: Tensor) -> Tensor:
    """Returns the indexes of the values in the input tensor."""
    indexes = []
    tensor = tensor.tolist()
    for value in values.tolist():
        indexes.append(tensor.index(value))
    return torch.tensor(indexes)


def _parse_sd(state_dict: OrderedDict) -> OrderedDict:
    """Parses state_dict to nested dictionaries."""
    parsed_sd = OrderedDict()
    for k, v in state_dict.items():
        _parse_param(k.split('.'), v, parsed_sd)
    return parsed_sd


def _parse_param(param: List, value: Tensor, dictionary: OrderedDict) -> None:
    """Parses value from state_dict to nested dictionaries."""
    if len(param) > 1:
        dictionary.setdefault(param[0], OrderedDict())
        _parse_param(param[1:], value, dictionary[param[0]])
    else:
        dictionary[param[0]] = value


def _collect_sd(parsed_state_dict: OrderedDict) -> OrderedDict:
    """Collect state_dict from nested dictionaries."""
    state_dict = OrderedDict()
    keys, values = _collect_param(parsed_state_dict)
    for k, v in zip(keys, values):
        key = '.'.join(k)
        state_dict[key] = v
    return state_dict


def _collect_param(dictionary: Union[OrderedDict, Tensor]) -> Tuple:
    """Collect value from nested dictionaries."""
    if isinstance(dictionary, OrderedDict):
        all_keys = []
        all_values = []
        for k, v in dictionary.items():
            keys, values = _collect_param(v)
            for key in keys:
                key.insert(0, k)
            all_values.extend(values)
            all_keys.extend(keys)
        return all_keys, all_values
    else:
        return [[]], [dictionary]


def _prune_resnet_block(block: Dict, input_channels: Tensor) -> Tensor:
    """Prune block of ResNet"""
    channels = input_channels
    downsample_channels = input_channels
    keys = list(block.keys())
    if 'downsample' in keys:
        filters = _check_nonzero_filters(block['downsample']['0']['weight'])
        block['downsample']['0']['weight'] = _prune_filters(
            weight=block['downsample']['0']['weight'],
            saving_filters=filters,
            saving_channels=downsample_channels
        )
        downsample_channels = filters
        block['downsample']['1'] = _prune_batchnorm(
            bn=block['downsample']['1'],
            saving_channels=downsample_channels
        )
        keys.remove('downsample')
    final_conv = keys[-2]
    final_bn = keys[-1]
    keys = keys[:-2]
    for key in keys:
        if key.startswith('conv'):
            filters = _check_nonzero_filters(block[key]['weight'])
            block[key]['weight'] = _prune_filters(
                weight=block[key]['weight'],
                saving_filters=filters,
                saving_channels=channels
            )
            channels = filters
        elif key.startswith('bn'):
            block[key] = _prune_batchnorm(bn=block[key], saving_channels=channels)
    filters = _check_nonzero_filters(block[final_conv]['weight'])
    filters = _index_union(filters, downsample_channels)
    block[final_conv]['weight'] = _prune_filters(
        weight=block[final_conv]['weight'],
        saving_filters=filters,
        saving_channels=channels,
    )
    channels = filters
    block[final_bn] = _prune_batchnorm(bn=block[final_bn], saving_channels=channels)
    block['indices'] = _indexes_of_tensor_values(channels, downsample_channels)
    return channels


def prune_resnet_state_dict(
        state_dict: OrderedDict,
) -> OrderedDict:
    """Prune state_dict of ResNet

    Args:
        state_dict: ``state_dict`` of ResNet model.

    Returns:
        Tuple(state_dict, input_channels, output_channels).
    """
    sd = _parse_sd(state_dict)
    filters = _check_nonzero_filters(sd['conv1']['weight'])
    sd['conv1']['weight'] = _prune_filters(
        weight=sd['conv1']['weight'], saving_filters=filters
    )
    channels = filters
    sd['bn1'] = _prune_batchnorm(bn=sd['bn1'], saving_channels=channels)

    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in sd[layer].values():
            channels = _prune_resnet_block(block=block, input_channels=channels)
    sd['fc']['weight'] = sd['fc']['weight'][:, channels].clone()
    sd = _collect_sd(sd)
    return sd


def sizes_from_state_dict(state_dict: OrderedDict) -> Dict:
    sd = _parse_sd(state_dict)
    sizes = {'conv1': sd['conv1']['weight'].shape}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        sizes[layer] = {}
        for i, block in enumerate(sd[layer].values()):
            sizes[layer][i] = {}
            for k, v in block.items():
                if k.startswith('conv'):
                    sizes[layer][i][k] = v['weight'].shape
                elif k == 'downsample':
                    sizes[layer][k] = v['0']['weight'].shape
                elif k == 'indices':
                    sizes[layer][i][k] = v.shape
    sizes['fc'] = sd['fc']['weight'].shape
    return sizes


def prune_resnet(model: ResNet) -> PrunedResNet:
    """Prune ResNet

    Args:
        model: ResNet model.

    Returns:
        Pruned ResNet model.

    Raises:
        AssertionError if model is not Resnet.
    """
    assert isinstance(model, ResNet), "Supports only ResNet models"
    model_type = MODELS_FROM_LENGHT[len(model.state_dict())]
    pruned_sd = prune_resnet_state_dict(model.state_dict())
    sizes = sizes_from_state_dict(pruned_sd)
    model = PRUNED_MODELS[model_type](sizes=sizes)
    model.load_state_dict(pruned_sd)
    return model


def load_sfp_resnet_model(
        state_dict_path: str,
) -> torch.nn.Module:
    """Loads SFP state_dict to PrunedResNet model.

    Args:
        state_dict_path: Path to state_dict file.

    Returns:
        PrunedResNet model.
    """
    state_dict = torch.load(state_dict_path, map_location='cpu')
    sizes = sizes_from_state_dict(state_dict)
    model_type = MODELS_FROM_LENGHT[len(list(filter((lambda x: not x.endswith('indices')), state_dict.keys())))]
    model = PRUNED_MODELS[model_type](sizes=sizes)
    model.load_state_dict(state_dict)
    return model

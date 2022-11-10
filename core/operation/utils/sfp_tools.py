from typing import List

import torch
from torch.linalg import vector_norm
from torch.nn import Conv2d, Parameter, Module


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


def prune_filters(
        conv: Conv2d,
        pruning_ratio: float,
        saving_channels: List[int],
) -> List[int]:
    """Prune filters of convolutional layer to the pruning_ratio (in-place)."""
    filter_pruned_num = int(conv.weight.size()[0] * pruning_ratio)
    filter_norms = vector_norm(conv.weight, dim=(1, 2, 3))
    _, indices = filter_norms.sort()
    w = conv.weight[indices[filter_pruned_num:]]
    conv.weight = Parameter(w[:, saving_channels].clone())
    conv.out_channels, conv.in_channels, _, _ = conv.weight.size()
    return indices[filter_pruned_num:]


def prune_batchnorm(bn: torch.nn.BatchNorm2d, saving_channels: List[int]):
    bn.bias = Parameter(bn.bias[saving_channels].clone())
    bn.weight = Parameter(bn.weight[saving_channels].clone())
    bn.running_mean = Parameter(bn.running_mean[saving_channels].clone())
    bn.running_var = Parameter(bn.running_var[saving_channels].clone())
    bn.num_features = len(saving_channels)


def prune_resnet(model: Module, pruning_ratio: float) -> None:
    channels = [0, 1, 2]
    for name, children in model.named_children():
        if isinstance(children, torch.nn.Conv2d):
            channels = prune_filters(children, pruning_ratio, channels)
        if isinstance(children, torch.nn.BatchNorm2d):
            prune_batchnorm(children, channels)

        if 'layer' in name:
            for block in children:
                block_in_channels = channels.clone()
                for n, c in block.named_children():
                    if isinstance(c, torch.nn.Conv2d):
                        channels = prune_filters(c, pruning_ratio, channels)
                    if isinstance(c, torch.nn.BatchNorm2d):
                        prune_batchnorm(c, channels)
                    if n == 'downsample':
                        cds = prune_filters(c[0], pruning_ratio, block_in_channels)
                        prune_batchnorm(c[1], cds)
        if name == 'fc':
            children.weight = Parameter(children.weight[:, channels].clone())

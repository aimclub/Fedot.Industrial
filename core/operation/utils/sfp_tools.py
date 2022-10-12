import torch
from torch.linalg import vector_norm
from torch.nn import Conv2d


def zerolize_filters(conv: Conv2d, pruning_ratio: float) -> None:
    """Zerolize filters of convolutional layer to the pruning_ratio (in-place)."""
    filter_pruned_num = int(conv.weight.size()[0] * pruning_ratio)
    filter_norms = vector_norm(conv.weight, dim=(1, 2, 3))
    _, indices = filter_norms.sort()
    with torch.no_grad():
        conv.weight[indices[:filter_pruned_num]] = 0

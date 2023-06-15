"""This module contains functions for working with singular value decomposition.

Model decomposition, pruning by threshold, decomposed model loading.
"""
from typing import Optional, Callable

import torch
from torch.nn.modules import Module
from torch.nn.modules.conv import Conv2d

from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d


def create_energy_svd_pruning(energy_threshold: float) -> Callable:
    """Returns the pruning function.
    Args:
        energy_threshold: pruning hyperparameter must be in the range (0, 1].
            the lower the threshold, the more singular values will be pruned.
    Returns:
        ``energy_svd_pruning`` function.
    Raises:
        Assertion Error: If ``energy_threshold`` is not in (0, 1].
    """
    assert 0 < energy_threshold <= 1, "energy_threshold must be in the range (0, 1]"
    def energy_svd_pruning(conv: DecomposedConv2d) -> None:
        """Prune the weight matrices to the energy_threshold (in-place).
        Args:
            conv: The optimizable layer.
        Raises:
            Assertion Error: If ``conv.decomposing`` is False.
        """
        assert conv.decomposing, "for pruning, the model must be decomposed"

        S, indices = conv.S.sort()
        U = conv.U[:, indices]
        Vh = conv.Vh[indices, :]
        sum = (S ** 2).sum()
        threshold = energy_threshold * sum
        for i, s in enumerate(S):
            sum -= s ** 2
            if sum < threshold:
                conv.set_U_S_Vh(U[:, i:].clone(), S[i:].clone(), Vh[i:, :].clone())
                break
    return energy_svd_pruning


def decompose_module(model: Module, decomposing_mode: Optional[str] = None) -> None:
    """Replace Conv2d layers with DecomposedConv2d layers in module (in-place).

    Args:
        model: Decomposable module.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
            If ``None`` replace layers without decomposition.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            decompose_module(module, decomposing_mode=decomposing_mode)

        if isinstance(module, Conv2d):
            new_module = DecomposedConv2d(
                base_conv=module,
                decomposing=False,
            )
            new_module.load_state_dict(module.state_dict())
            if decomposing_mode is not None:
                new_module.decompose(decomposing_mode=decomposing_mode)
            setattr(model, name, new_module)


def _load_svd_params(model, state_dict, prefix='') -> None:
    """Loads state_dict to DecomposedConv2d layers in model."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _load_svd_params(module, state_dict, prefix=f'{prefix}{name}.')

        if isinstance(module, DecomposedConv2d):
            module.set_U_S_Vh(
                u=state_dict[f'{prefix}{name}.U'],
                s=state_dict[f'{prefix}{name}.S'],
                vh=state_dict[f'{prefix}{name}.Vh']
            )


def load_svd_state_dict(
        model: Module,
        decomposing_mode: str,
        state_dict_path: str
) -> None:
    """Loads SVD state_dict to model.

    Args:
        model: An instance of the base model.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        state_dict_path: Path to state_dict file.
    """
    state_dict = torch.load(state_dict_path, map_location='cpu')
    decompose_module(model=model, decomposing_mode=decomposing_mode)
    _load_svd_params(model, state_dict)
    model.load_state_dict(state_dict)

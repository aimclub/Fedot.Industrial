import pytest
import random
import torch
from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d


def run(mode):
    for in_channels in [3, 32]:
        for kernel_size in [(3, 5), (7, 1)]:
            for stride in [(1, 2), (2, 3)]:
                for padding in [(1, 2), (2, 3)]:
                    for dilation in [(1, 2), (2, 3)]:
                        base_conv = torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=32,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                        )
                        dconvs = {
                            'dconv': DecomposedConv2d(base_conv, None),
                            'one_layer': DecomposedConv2d(base_conv, mode),
                            'two_layers': DecomposedConv2d(base_conv, mode, forward_mode='two_layers'),
                            'three_layers': DecomposedConv2d(base_conv, mode, forward_mode='three_layers')
                        }
                        x = torch.rand((random.randint(1, 16), in_channels, random.randint(28, 1000), random.randint(28, 1000)))
                        y_true = base_conv(x)
                        for name, dconv in dconvs.items():
                            y = dconv(x)
                            is_ok = torch.allclose(y, y_true, rtol=0.0001, atol=0.00001)
                            print(is_ok)
                            assert is_ok, f"{mode}: {base_conv} {torch.isclose(y, y_true)}"


def test_channel_decomposed_conv():
    run('channel')


def test_spatial_decomposed_conv():
    run('spatial')

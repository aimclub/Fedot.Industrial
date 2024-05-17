import pytest
import random
import torch
from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d


@pytest.fixture(scope='module')
def params():
    return dict(in_channels=3,
                out_channels=32,
                kernel_size=(3, 5),
                stride=(1, 2),
                padding=(1, 2),
                dilation=(1, 2))


def run(mode, params):
    base_conv = torch.nn.Conv2d(
        in_channels=params['in_channels'],
        out_channels=params['out_channels'],
        kernel_size=params['kernel_size'],
        stride=params['stride'],
        padding=params['padding'],
        dilation=params['dilation'],
    )
    dconvs = {
        'dconv': DecomposedConv2d(base_conv, None),
        'one_layer': DecomposedConv2d(base_conv, mode),
        'two_layers': DecomposedConv2d(base_conv, mode, forward_mode='two_layers'),
        'three_layers': DecomposedConv2d(base_conv, mode, forward_mode='three_layers')
    }
    x = torch.rand(
        (random.randint(
            1, 16), params['in_channels'], random.randint(
            28, 1000), random.randint(
                28, 1000)))
    y_true = base_conv(x)
    for name, dconv in dconvs.items():
        y = dconv(x)
        is_ok = torch.allclose(y, y_true, rtol=0.0001, atol=0.00001)
        print(is_ok)
        assert is_ok, f"{mode}: {base_conv} {torch.isclose(y, y_true)}"


def test_channel_decomposed_conv(params):
    run('channel', params)


def test_spatial_decomposed_conv(params):
    run('spatial', params)

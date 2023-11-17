import torch
import torch.nn as nn
from torch.autograd import Variable
from fedot_ind.core.models.cnn.unet import DoubleConv, Down, Up, OutConv, UNet


def test_double_conv():
    in_channels, out_channels = 3, 64
    x = Variable(torch.rand((1, in_channels, 128, 128)))
    double_conv = DoubleConv(in_channels, out_channels)
    output = double_conv(x)
    assert output.size() == torch.Size([1, out_channels, 128, 128])


def test_down():
    in_channels, out_channels = 64, 128
    x = Variable(torch.rand((1, in_channels, 128, 128)))
    down = Down(in_channels, out_channels)
    output = down(x)
    assert output.size() == torch.Size([1, out_channels, 64, 64])


def test_up():
    in_channels, out_channels = 128, 64
    x1 = Variable(torch.rand((1, in_channels, 64, 64)))
    x2 = Variable(torch.rand((1, in_channels, 128, 128)))
    up = Up(in_channels, out_channels)


def test_out_conv():
    in_channels, out_channels = 64, 3
    x = Variable(torch.rand((1, in_channels, 128, 128)))
    out_conv = OutConv(in_channels, out_channels)
    output = out_conv(x)
    assert output.size() == torch.Size([1, out_channels, 128, 128])


def test_unet():
    n_channels, n_classes = 3, 3
    x = Variable(torch.rand((1, n_channels, 128, 128)))
    unet = UNet(n_channels, n_classes)
    output = unet(x)['out']
    assert output.size() == torch.Size([1, n_classes, 128, 128])

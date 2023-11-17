import pytest
import torch

from fedot_ind.core.models.nn.inception import Inception, InceptionBlock, InceptionTimeNetwork, InceptionTranspose

in_channels = 3
out_channels = 32
n_filters = 16
batch_size = 32
sequence_length = 64


@pytest.fixture
def tensor():
    return torch.randn((batch_size, in_channels, sequence_length))


def test_inception(tensor):
    inception = Inception(in_channels=in_channels,
                          n_filters=n_filters,
                          return_indices=True)
    result = inception.forward(X=tensor)
    assert result is not None


def test_inception_block(tensor):
    inception_block = InceptionBlock(in_channels=3,
                                     n_filters=32,
                                     return_indices=True)
    result = inception_block.forward(X=tensor)
    assert inception_block is not None


def test_inception_transpose():
    inception_transpose = InceptionTranspose(in_channels=in_channels,
                                             out_channels=out_channels)
    assert inception_transpose is not None


def test_InceptionTransposeBlock():
    inception_transpose_block = InceptionTranspose(in_channels=in_channels,
                                                   out_channels=out_channels)
    assert inception_transpose_block is not None


def test_InceptionTimeNetwork():
    inception_time_network = InceptionTimeNetwork({})
    architecture = inception_time_network.network_architecture
    assert inception_time_network is not None
    assert architecture is not None
import math
from typing import Optional

import torch
from fastai.layers import SigmoidRange
from fastai.torch_core import Module
from torch import nn, Tensor

from fedot_ind.core.architecture.settings.computational import default_device
from fedot_ind.core.models.nn.network_modules.layers.attention_layers import \
    ScaledDotProductAttention, \
    MultiHeadAttention
from fedot_ind.core.models.nn.network_modules.layers.conv_layers import Conv1d
from fedot_ind.core.models.nn.network_modules.layers.linear_layers import Transpose, Flatten
from fedot_ind.core.models.nn.network_modules.layers.padding_layers import Pad1d
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn


class _TSTEncoderLayer(Module):
    def __init__(self,
                 q_len: int,
                 d_model: int,
                 n_heads: int,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        assert d_model // n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        if d_k is None:
            d_k = d_model // n_heads
        if d_v is None:
            d_v = d_model // n_heads

        # Multi-Head attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        self.batchnorm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.batchnorm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Multi-Head attention sublayer
        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src, mask=mask)
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        src = self.batchnorm_attn(src)  # Norm: batchnorm

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        src = self.batchnorm_ffn(src)  # Norm: batchnorm

        return src


class _TSTEncoder(Module):
    def __init__(self,
                 q_len,
                 d_model,
                 n_heads,
                 d_k=None,
                 d_v=None,
                 d_ff=None,
                 dropout=0.1,
                 activation='gelu',
                 n_layers=1):
        self.layers = nn.ModuleList(
            [_TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout,
                              activation=activation) for i in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers: output = mod(output)
        return output


class TST(Module):
    def __init__(self, c_in: int,
                 c_out: int,
                 seq_len: int,
                 max_seq_len: Optional[int] = None,
                 n_layers: int = 3,
                 d_model: int = 128,
                 n_heads: int = 16,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 act: str = "gelu",
                 fc_dropout: float = 0.,
                 y_range: Optional[tuple] = None,
                 verbose: bool = False, **kwargs):
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            n_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            dropout: amount of residual dropout applied in the encoder.
            act: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.

        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        self.c_out, self.seq_len = c_out, seq_len

        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        if max_seq_len is not None and seq_len > max_seq_len:  # Control temporal resolution
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(Pad1d(padding),
                                     Conv1d(c_in, d_model, kernel_size=tr_factor, padding=0, stride=tr_factor))
            print(
                f'temporal resolution modified: {seq_len} --> {q_len} time steps: kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n',
                verbose)
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs)  # Eq 2
            print(f'Conv1d with kwargs={kwargs} applied to input to create input encodings\n', verbose)
        else:
            self.W_P = nn.Linear(c_in, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        W_pos = torch.empty((q_len, d_model), device=default_device())
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout,
                                   activation=act, n_layers=n_layers)
        self.flatten = Flatten()

        # Head
        self.head_nf = q_len * d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None, **kwargs):
        layers = [get_activation_fn(act), Flatten()]
        if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        layers += [nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # x: [bs x nvars x q_len]

        # Input encoding
        if self.new_q_len:
            u = self.W_P(x).transpose(2,
                                      1)  # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else:
            u = self.W_P(x.transpose(2,
                                     1))  # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)  # z: [bs x q_len x d_model]
        z = z.transpose(2, 1).contiguous()  # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.head(z)

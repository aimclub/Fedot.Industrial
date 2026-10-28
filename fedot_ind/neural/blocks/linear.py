"""Small head blocks with stable sequential state-dict names."""
from torch import nn


class LinLnDrop(nn.Sequential):
    def __init__(self, n_in, n_out, ln=True, p=0., act=None, lin_first=False):
        layers = [nn.LayerNorm(n_out if lin_first else n_in)] if ln else []
        if p != 0:
            layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not ln)]
        if act is not None:
            lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)

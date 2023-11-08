from fedot_ind.core.models.nn.modules import *
import matplotlib as plt
from fastai.torch_core import Module


def get_acts_and_grads(model,
                       modules,
                       x,
                       y=None,
                       detach=True,
                       cpu=False):
    r"""Returns activations and gradients for given modules in a model and a single input or a batch.
    Gradients require y value(s). If they are not provided, it will use the predictions. """
    if not is_listy(modules): modules = [modules]
    x = x[None, None] if x.ndim == 1 else x[None] if x.ndim == 2 else x
    if cpu:
        model = model.cpu()
        x = x.cpu()
    with hook_outputs(modules, detach=detach, cpu=cpu) as h_act:
        with hook_outputs(modules, grad=True, detach=detach, cpu=cpu) as h_grad:
            preds = model.eval()(x)
            if y is None:
                preds.max(dim=-1).values.mean().backward()
            else:
                y = y.detach().cpu().numpy()
                if preds.shape[0] == 1:
                    preds[0, y].backward()
                else:
                    if y.ndim == 1: y = y.reshape(-1, 1)
                    torch_slice_by_dim(preds, y).mean().backward()
    if len(modules) == 1:
        return h_act.stored[0].data, h_grad.stored[0][0].data
    else:
        return [h.data for h in h_act.stored], [h[0].data for h in h_grad.stored]


def get_attribution_map(model, modules, x, y=None, detach=True, cpu=False, apply_relu=True):
    def _get_attribution_map(A_k, w_ck):
        dim = (0, 2, 3) if A_k.ndim == 4 else (0, 2)
        w_ck = w_ck.mean(dim, keepdim=True)
        L_c = (w_ck * A_k).sum(1)
        if apply_relu: L_c = nn.ReLU()(L_c)
        if L_c.ndim == 3:
            return L_c.squeeze(0) if L_c.shape[0] == 1 else L_c
        else:
            return L_c.repeat(x.shape[1], 1) if L_c.shape[0] == 1 else L_c.unsqueeze(1).repeat(1, x.shape[1], 1)

    if x.ndim == 1:
        x = x[None, None]
    elif x.ndim == 2:
        x = x[None]
    A_k, w_ck = get_acts_and_grads(model, modules, x, y, detach=detach, cpu=cpu)
    if is_listy(A_k):
        return [_get_attribution_map(A_k[i], w_ck[i]) for i in range(len(A_k))]
    else:
        return _get_attribution_map(A_k, w_ck)


class XCM(Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 seq_len: Optional[int] = None,
                 nf: int = 128,
                 window_perc: float = 1.,
                 flatten: bool = False,
                 custom_head: callable = None,
                 concat_pool: bool = False,
                 fc_dropout: float = 0.,
                 bn: bool = False,
                 y_range: tuple = None,
                 **kwargs):

        window_size = int(round(seq_len * window_perc, 0))
        self.conv2dblock = nn.Sequential(
            *[Unsqueeze(1), Conv2d(1, nf, kernel_size=(1, window_size), padding='same'), BatchNorm(nf), nn.ReLU()])
        self.conv2d1x1block = nn.Sequential(*[nn.Conv2d(nf, 1, kernel_size=1), nn.ReLU(), Squeeze(1)])
        self.conv1dblock = nn.Sequential(
            *[Conv1d(c_in, nf, kernel_size=window_size, padding='same'), BatchNorm(nf, ndim=1), nn.ReLU()])
        self.conv1d1x1block = nn.Sequential(*[nn.Conv1d(nf, 1, kernel_size=1), nn.ReLU()])
        self.concat = Concat()
        self.conv1d = nn.Sequential(
            *[Conv1d(c_in + 1, nf, kernel_size=window_size, padding='same'), BatchNorm(nf, ndim=1), nn.ReLU()])

        self.head_nf = nf
        self.c_out = c_out
        self.seq_len = seq_len
        if custom_head:
            self.head = custom_head(self.head_nf, c_out, seq_len, **kwargs)
        else:
            self.head = self.create_head(self.head_nf, c_out, seq_len, flatten=flatten, concat_pool=concat_pool,
                                         fc_dropout=fc_dropout, bn=bn, y_range=y_range)

    def forward(self, x):
        x1 = self.conv2dblock(x)
        x1 = self.conv2d1x1block(x1)
        x2 = self.conv1dblock(x)
        x2 = self.conv1d1x1block(x2)
        out = self.concat((x2, x1))
        out = self.conv1d(out)
        out = self.head(out)
        return out

    def create_head(self, nf, c_out, seq_len=None, flatten=False, concat_pool=False, fc_dropout=0., bn=False,
                    y_range=None):
        if flatten:
            nf *= seq_len
            layers = [Reshape()]
        else:
            if concat_pool: nf *= 2
            layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

    def show_gradcam(self,
                     x,
                     y=None,
                     detach=True,
                     cpu=True,
                     apply_relu=True,
                     cmap='inferno',
                     figsize=None,
                     **kwargs):

        att_maps = get_attribution_map(self, [self.conv2dblock, self.conv1dblock], x, y=y, detach=detach, cpu=cpu,
                                       apply_relu=apply_relu)
        att_maps[0] = (att_maps[0] - att_maps[0].min()) / (att_maps[0].max() - att_maps[0].min())
        att_maps[1] = (att_maps[1] - att_maps[1].min()) / (att_maps[1].max() - att_maps[1].min())

        if figsize is None:
            figsize = (10, 10)

        fig = plt.figure(figsize=figsize, **kwargs)
        ax = plt.axes()
        plt.title('Observed variables')
        if att_maps[0].ndim == 3:
            att_maps[0] = att_maps[0].mean(0)
        im = ax.imshow(att_maps[0], cmap=cmap)
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        plt.colorbar(im, cax=cax)
        plt.show()

        fig = plt.figure(figsize=figsize, **kwargs)
        ax = plt.axes()
        plt.title('Time')
        if att_maps[1].ndim == 3:
            att_maps[1] = att_maps[1].mean(0)
        im = ax.imshow(att_maps[1], cmap=cmap)
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        plt.colorbar(im, cax=cax)
        plt.show()

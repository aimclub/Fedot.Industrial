import numpy as np
import torch
from spectrum import speriodogram, pev, parma
from fedot_ind.core.operation.transformation.torch_specter.speriodogram import speriodogram_torch, get_window_torch
from fedot_ind.core.operation.transformation.torch_specter.eigen import pev_torch
from spectrum.window import Window


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))



def compare_periodogram_rmse():
    np.random.seed(0)
    x = np.random.randn(100)
    # x = np.random.randn(2, 5000).astype(np.float32).squeeze()
    windows = ["kaiser", "hamming", "hann", "blackman",] 
    # for w in windows:
    #     w1 = get_window_torch(w, x.shape[0])
    #     print(w1)
    #     w2 = Window(x.shape[0], w)
    #     print(w, rmse(np.array(w1), np.array(w2.data)))

    psd_np = speriodogram(
        x,
        sampling=1.0,
        window="hamming",
        detrend=False,
        scale_by_freq=True,
        NFFT=1000,
    )

    psd_torch = speriodogram_torch(
        torch.tensor(x),
        sampling=1.0,
        window="hamming",
        detrend=False,
        NFFT=1000,
        scale_by_freq=True,
    ).detach().cpu().numpy()
    
    print("RMSE:", rmse(psd_np, psd_torch))
    print("Max abs diff:", np.max(np.abs(psd_np - psd_torch)))

def compare_eigen_rmse():
    np.random.seed(0)
    x = np.random.randn(100)

    psd_np = pev(
        x,
        IP=10,
        # sampling=1.0,
        # scale_by_freq=False,
        # NFFT=100,
    ).psd

    psd_torch = pev_torch(
        torch.tensor(x),
        10,
        # sampling=1.0,
        NFFT=None,
        # scale_by_freq=False,
    ).detach().cpu().numpy()

    print("RMSE:", rmse(psd_np, psd_torch))
    print("Max abs diff:", np.max(np.abs(psd_np - psd_torch)))


if __name__ == "__main__":
    compare_eigen_rmse()
    compare_periodogram_rmse()

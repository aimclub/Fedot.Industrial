import torch
import math


def get_window_torch(window, N, device=None, dtype=torch.float32):
    """
    Create window function in Torch
    """
    n = torch.arange(N, device=device, dtype=dtype)

    if window in (None, 'rectangular'):
        return torch.ones(N, device=device, dtype=dtype)

    if window == 'hann':
        return 0.5 - 0.5 * torch.cos(2 * math.pi * n / (N - 1))

    if window == 'hamming':
        return 0.54 - 0.46 * torch.cos(2 * math.pi * n / (N - 1))

    if window == 'blackman':
        return (
            0.42
            - 0.5 * torch.cos(2 * math.pi * n / (N - 1))
            + 0.08 * torch.cos(4 * math.pi * n / (N - 1))
        )

    if window == 'kaiser':
        beta = 8.6  # стандартное значение
        return torch.kaiser_window(N, periodic=False, beta=beta, device=device, dtype=dtype)

    raise ValueError(f"Unsupported window type: {window}")


def speriodogram_torch(
    x,
    NFFT=None,
    detrend=None,
    sampling=4096,
    scale_by_freq=True,
    window='hamming',
    axis=0,
):
    x = torch.as_tensor(x, dtype=torch.float64)

    # 1D / 2D handling
    if x.ndim == 1:
        r = x.shape[0]
        axis = 0
        w = get_window_torch(window, r, device=x.device, dtype=x.dtype)
    elif x.ndim == 2:
        r, c = x.shape
        w = torch.stack(
            [get_window_torch(window, r, device=x.device, dtype=x.dtype)
             for _ in range(c)],
            dim=1
        )
    else:
        raise ValueError("x must be 1D or 2D")

    if NFFT is None:
        NFFT = r

    # detrend
    if detrend is True:
        m = torch.mean(x, dim=axis, keepdim=True)
    else:
        m = 0.0

    xw = x * w - m

    # FFT
    isreal = torch.isreal(x).all()

    if isreal:
        if x.ndim == 2:
            psd = torch.abs(
                torch.fft.rfft(xw, n=NFFT, dim=0)
            ) ** 2 / r
        else:
            psd = torch.abs(
                torch.fft.rfft(xw, n=NFFT, dim=-1)
            ) ** 2 / r
    else:
        if x.ndim == 2:
            psd = torch.abs(
                torch.fft.fft(xw, n=NFFT, dim=0)
            ) ** 2 / r
        else:
            psd = torch.abs(
                torch.fft.fft(xw, n=NFFT, dim=-1)
            ) ** 2 / r
    
    

    # scale by frequency
    if scale_by_freq:
        df = sampling / float(NFFT)
        psd = psd * (2 * math.pi / df)

    return psd.T if x.ndim == 1 else psd

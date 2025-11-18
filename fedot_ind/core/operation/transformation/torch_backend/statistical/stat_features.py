import warnings
from scipy.stats import skew as skw, kurtosis as kurt
import torch

warnings.filterwarnings("ignore")


def mean_torch(x: torch.Tensor, axis=-1):
    mean = torch.mean(x, axis=axis)
    return mean.item() if mean.numel() == 1 else mean


def median_torch(x: torch.Tensor, axis=-1):
    x_sorted, _ = torch.sort(x, dim=axis)
    n = x.size(axis)
    if n % 2 == 1:
        m = x_sorted.select(axis, n // 2)
    else:
        m = (x_sorted.select(axis, n // 2 - 1) +
             x_sorted.select(axis, n // 2)) / 2
    return m.item() if m.numel() == 1 else m


def std_torch(x: torch.Tensor, axis=-1):
    std = torch.std(x, axis=axis, unbiased=False)
    return std.item() if std.numel() == 1 else std


def max_torch(x: torch.Tensor, axis=-1):
    max = torch.max(x, axis=axis).values
    return max.item() if max.numel() == 1 else max


def min_torch(x: torch.Tensor, axis=-1):
    min = torch.min(x, axis=axis).values
    return min.item() if min.numel() == 1 else min


def q5_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.05, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def q25_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.25, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def q75_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.75, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def q95_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.95, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def lambda_less_zero(array: torch.Tensor, axis=-1) -> int | torch.Tensor:
    mask = (array < 0.01).int()
    return torch.sum(mask).item() if mask.numel() == 1 else torch.sum(mask, dim=axis)


def quantile_torch(array: torch.Tensor, q: float, axis=-1) -> float | torch.Tensor:
    axis = axis % array.ndim
    quant = torch.quantile(input=array, q=q, dim=axis)
    return quant.item() if quant.numel() == 1 else quant


def diff(array: torch.Tensor, axis=-1) -> float:
    return (array[-1] - array[0]).item()


def skewness_torch(array: torch.Tensor, axis=-1):
    result = skw(a=array, axis=axis)
    return result if array.ndim == 1 else torch.Tensor(skw(result))


def kurtosis_torch(array: torch.Tensor, axis=None) -> float:
    result = kurt(a=array, axis=axis)
    return result if array.ndim == 1 else torch.Tensor(skw(result))


def n_peaks_torch(X: torch.Tensor, axis=-1) -> int:
    if (axis != -1) | (axis != len(X.shape)):
        x = X.transpose(axis, -1)
    if X.ndim > 2:
        x = X.reshape(-1, x.shape[-1])
    if X.ndim == 1:
        x = X.unsqueeze(0)
    mid = x[:, 1:-1]
    left = x[:, :-2]
    right = x[:, 2:]
    peaks_mask = ((mid > left) & (mid > right))
    n_peaks = torch.sum(peaks_mask, dim=-1).to(torch.float32)
    return n_peaks.reshape(X.shape[0], X.shape[1]) if n_peaks.numel() > 1 else n_peaks.item()


def mean_ptp_distance_torch(X: torch.Tensor, axis=-1):
    if (axis != -1) | (axis != len(X.shape)):
        x = X.transpose(axis, -1)
    if X.ndim > 2:
        x = X.reshape(-1, x.shape[-1])
    if X.ndim == 1:
        x = X.unsqueeze(0)
    T = x.shape[0]

    mid = x[:, 1:-1]
    left = x[:, :-2]
    right = x[:, 2:]
    peaks_mask = ((mid > left) & (mid > right))

    mean_ptp = torch.zeros(T)
    for i in range(T):
        peak_indices = torch.nonzero(peaks_mask[i]).squeeze() + 1
        if peak_indices.numel() > 1:
            diffs = torch.diff(peak_indices.to(torch.float32))
            mean_ptp[i] = torch.mean(diffs).item()
    return mean_ptp.reshape(X.shape[0], X.shape[1]) if mean_ptp.numel() > 1 else mean_ptp.item()


def slope_plural_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    y = array.to(torch.float32)
    axis = axis % y.ndim
    n = y.shape[axis]
    x = torch.arange(n, device=y.device, dtype=torch.float32)
    x_mean = x.mean()
    y_mean = y.mean(dim=axis, keepdim=True)
    slope = torch.sum((x - x_mean) * (y - y_mean), dim=axis) / torch.sum((x - x_mean) ** 2)
    return slope.item() if slope.numel() == 1 else slope


def ben_corr_torch(x: torch.Tensor, axis: int = -1):
    """
    Vectorized Benford correlation for multi-dimensional tensors.
    Correlation is computed along `axis` (default: last axis).

    Output shape = x.shape without the `axis` dimension.
    """
    if (axis != -1) | (axis != len(x.shape)):
        x = x.transpose(axis, -1)
    *batch_shape, T = x.shape
    B = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
    x_flat = x.reshape(B, T)
    # preprocess and get histogramm of first digits for each ts
    x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
    x_flat = torch.abs(x_flat)
    x_flat = torch.clamp(x_flat, min=1e-8)
    exponents = torch.floor(torch.log10(x_flat))
    mantissas = x_flat / (10**exponents)
    first_digits = torch.floor(mantissas).clamp(1, 9).to(torch.int64)
    offsets = torch.arange(B, device=x.device) * 10
    fd_flat = (first_digits + offsets.unsqueeze(1)).reshape(-1)
    counts = torch.bincount(fd_flat, minlength=B * 10)
    counts = counts.reshape(B, 10)[:, 1:10]
    data_dist = counts / (counts.sum(dim=1, keepdim=True) + 1e-8)
    # Benford's distribution
    digits = torch.arange(1, 10, device=x.device, dtype=torch.float32)
    benford = torch.log10(1 + 1 / digits).unsqueeze(0).expand(B, -1)
    # corr coef
    x_mean = benford.mean(dim=1, keepdim=True)
    y_mean = data_dist.mean(dim=1, keepdim=True)
    num = ((benford - x_mean) * (data_dist - y_mean)).sum(dim=1)
    den = torch.sqrt(
        ((benford - x_mean)**2).sum(dim=1) *
        ((data_dist - y_mean)**2).sum(dim=1)
    )
    corr = num / (den + 1e-12)
    corr = torch.nan_to_num(corr, nan=0.0)
    return corr.reshape(batch_shape) if batch_shape else corr.item()


def interquantile_range_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    return quantile_torch(array, 0.75, axis) - quantile_torch(array, 0.25, axis)


def energy_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    axis = axis % array.ndim
    energy = torch.sum(array ** 2, dim=axis) / array.shape[axis]
    return energy.item() if energy.numel() == 1 else energy


def autocorrelation_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    axis = axis % x.ndim
    lagged = torch.roll(x, shifts=1, dims=axis)
    x_centered = x - x.mean(dim=axis, keepdim=True)
    lagged_centered = lagged - lagged.mean(dim=axis, keepdim=True)
    num = (x_centered * lagged_centered).sum(dim=axis)
    denom = torch.sqrt((x_centered ** 2).sum(dim=axis) * (lagged_centered ** 2).sum(dim=axis))
    corr = num / denom.clamp(min=1e-12)
    return corr.item() if corr.numel() == 1 else corr


def zero_crossing_rate_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if (axis != -1) | (axis != len(x.shape)):
        x = x.transpose(axis, -1)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    x_min = x.min(dim=axis, keepdim=True).values
    x_max = x.max(dim=axis, keepdim=True).values
    x_scaled = 2 * (x - x_min) / (x_max - x_min + 1e-12) - 1
    signs = torch.sign(x_scaled)
    signs[signs == 0] = -1
    diff = torch.diff(signs, dim=axis)
    crossings = (diff != 0).sum(dim=axis)
    rate = crossings / x.shape[axis]
    return rate.item() if rate.numel() == 1 else rate


def shannon_entropy_torch(x: torch.Tensor, axis=None) -> float | torch.Tensor:
    """Returns the Shannon Entropy of the time series.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    entropy = []
    for arr in x:
        _, counts = torch.unique(arr, return_counts=True)
        p = counts.float() / counts.sum()
        entropy.append(-(p * torch.log2(p + 1e-12)).sum())
    entropy = torch.stack(entropy)
    return entropy.item() if entropy.numel() == 1 else entropy


def base_entropy(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    axis = axis % x.ndim
    probs = x / (x.sum(dim=axis, keepdim=True) + 1e-12)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=axis)
    return entropy.item() if entropy.numel() == 1 else entropy


def ptp_amp_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if axis < 0:
        axis = x.ndim + axis
    diff = x.max(dim=axis).values - x.min(dim=axis).values
    return diff.item() if diff.numel() == 1 else diff


def crest_factor_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if not torch.is_floating_point(x):
        x = x.float()
    num = x.abs().max(dim=axis).values
    den = torch.sqrt((x ** 2).mean(dim=axis)) + 1e-12
    res = num / den
    return res.item() if res.numel() == 1 else res


def mean_ema_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    """Calculate weights before ema, not itteratively.
    """
    T = x.shape[axis]
    span = max(int(T / 10), 2)
    alpha = 2 / (span + 1)
    weights = (1 - alpha) ** torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    ema = torch.sum(x * weights, dim=axis)
    return ema.item() if ema.numel() == 1 else ema


def mean_moving_median_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    T = x.shape[-1]
    span = max(int(T / 10), 2)
    medians = []
    for i in range(T - span + 1):
        window = x[..., i:i + span]
        median = median_torch(window)
        median = torch.Tensor([median]) if isinstance(median, float) else median
        medians.append(median)
        # medians.append(window.median(dim=-1).values)
    res = torch.stack(medians, dim=-1).mean(dim=-1)
    return res.item() if res.numel() == 1 else res


def hjorth_mobility_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if not torch.is_floating_point(x):
        x = x.float()
    diff = torch.diff(x, dim=axis)
    M2 = (diff ** 2).mean(dim=axis)
    TP = (x ** 2).mean(dim=axis)
    mobility = torch.sqrt(M2 / (TP + 1e-12))
    return mobility.item() if mobility.numel() == 1 else mobility


def hjorth_complexity_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if not torch.is_floating_point(x):
        x = x.float()
    diff = torch.diff(x, dim=axis)
    M2 = (diff ** 2).mean(dim=axis)
    TP = (x ** 2).mean(dim=axis)
    M4 = (torch.diff(diff, dim=axis) ** 2).mean()
    complexity = torch.sqrt((M4 * TP) / (M2 * M2 + 1e-12))
    return complexity.item() if complexity.numel() == 1 else complexity


def hurst_exponent_torch(X: torch.Tensor, axis: int = -1) -> torch.Tensor:
    if (axis != -1) | (axis != len(X.shape)):
        x = X.transpose(axis, -1)
    if X.ndim > 2:
        x = X.reshape(-1, x.shape[-1])
    if X.ndim == 1:
        x = X.unsqueeze(0)

    B, T = x.shape
    device = x.device
    dtype = x.dtype

    t = torch.arange(1, T + 1, device=device, dtype=dtype).unsqueeze(0)
    y = torch.cumsum(x, dim=-1)
    ave_t = y / t
    S_T = torch.zeros((B, T), device=device, dtype=dtype)
    R_T = torch.zeros((B, T), device=device, dtype=dtype)
    for i in range(T):
        segment = x[:, :i + 1]
        S_T[:, i] = segment.std(dim=-1, unbiased=False)
        X_T = y - t * ave_t[:, i].unsqueeze(1)
        R_T[:, i] = X_T[:, :i + 1].amax(dim=-1) - X_T[:, :i + 1].amin(dim=-1)
    RS = R_T / (S_T + 1e-8)
    RS = torch.log(RS[:, 1:])
    n = torch.log(t[:, 1:]).squeeze(0)

    ones = torch.ones_like(n)
    A = torch.stack([n, ones], dim=1)
    H = torch.empty(B, device=device, dtype=dtype)
    for b in range(B):
        sol = torch.linalg.lstsq(A, RS[b].unsqueeze(1)).solution.squeeze()
        H[b] = sol[0]
    if X.ndim > 2:
        H = H.reshape(X.shape[0], X.shape[1])
    return H if H.numel() > 1 else H.item()


def pfd_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)

    D = torch.diff(x, dim=axis)
    N_delta = ((D[..., 1:] * D[..., :-1]) < 0).sum(dim=axis)
    n = x.shape[axis]
    n = torch.tensor(float(n), device=x.device)
    num = torch.log10(n)
    den = torch.log10(n) + torch.log10(n / (n + 0.4 * N_delta))
    res = num / den
    return res.item() if res.numel() == 1 else res

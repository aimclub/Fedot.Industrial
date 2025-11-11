import warnings
from scipy.stats import skew as skw, kurtosis as kurt
import torch

warnings.filterwarnings("ignore")


def q5_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.05, dim=axis)
    return quant.item() if  quant.numel() == 1 else quant


def q25_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.25, dim=axis)
    return quant.item() if  quant.numel() == 1 else quant


def q75_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.75, dim=axis)
    return quant.item() if  quant.numel() == 1 else quant


def q95_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    quant = torch.quantile(input=array, q=0.95, dim=axis)
    return quant.item() if  quant.numel() == 1 else quant


def lambda_less_zero(array: torch.Tensor, axis=None) -> int | torch.Tensor:
    mask = (array < 0.01).int()
    return torch.sum(mask).item() if mask.numel() == 1 else torch.sum(mask, dim=axis)


def quantile_torch(array: torch.Tensor, q: float, axis=-1) -> float | torch.Tensor:
    axis = axis % array.ndim
    quant = torch.quantile(input=array, q=q, dim=axis)
    return quant.item() if  quant.numel() == 1 else quant


def diff(array: torch.Tensor, axis=None) -> float:
    return (array[-1] - array[0]).item()


def skewness_torch(array: torch.Tensor, axis=None) -> float | torch.Tensor:
    return skw(a=array) if axis is None else torch.Tensor(skw(a=array, axis=axis))


def kurtosis_torch(array: torch.Tensor, axis=None) -> float:
    return kurt(a=array) if axis is None else torch.Tensor(kurt(a=array, axis=axis))


def n_peaks_torch(array: torch.Tensor, axis=None) -> int:
    if array.ndim > 1:
        return None
    else:
        peaks_mask = (array[1:-1] > array[:-2]) & (array[1:-1] > array[2:])
        return int(torch.sum(peaks_mask).item())


def mean_ptp_distance_torch(array: torch.Tensor, axis=None):
    if array.ndim > 1:
        return None
    else:
        peaks_mask = (array[1:-1] > array[:-2]) & (array[1:-1] > array[2:])
        peak_indices = torch.nonzero(peaks_mask).squeeze() + 1
        if peak_indices.numel() < 2:
            return 0.0
        else:
            diffs = torch.diff(peak_indices.to(torch.float32))
            return torch.mean(diffs).item()


def slope_plural_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    y = array.to(torch.float32)
    axis = axis % y.ndim
    n = y.shape[axis]
    x = torch.arange(n, device=y.device, dtype=torch.float32)
    x_mean = x.mean()
    y_mean = y.mean(dim=axis, keepdim=True)
    slope = torch.sum((x - x_mean) * (y - y_mean), dim=axis) / torch.sum((x - x_mean) ** 2)
    return slope.item() if slope.numel() == 1 else slope


def ben_corr_torch(x: torch.Tensor, axis=None) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.abs(x)
    x = torch.where(x == 0, torch.tensor(1e-8, device=x.device), x)
    exponents = torch.floor(torch.log10(x))
    mantissas = x / (10 ** exponents)
    first_digits = torch.floor(mantissas).clamp(1, 9).to(torch.int64)
    B = first_digits.shape[0]
    offsets = torch.arange(B, device=x.device) * 10
    fd_flat = (first_digits + offsets.unsqueeze(1)).flatten()
    counts = torch.bincount(fd_flat, minlength=B * 10)
    counts = counts.reshape(B, 10)
    counts = counts[:, 1:10]
    data_distribution = counts / counts.sum(dim=1, keepdim=True)
    digits = torch.arange(1, 10, device=x.device, dtype=torch.float32)
    benford_distribution = torch.log10(1 + 1 / digits)
    benford_distribution = benford_distribution.unsqueeze(0).expand(B, -1)
    x_mean = benford_distribution.mean(dim=1, keepdim=True)
    y_mean = data_distribution.mean(dim=1, keepdim=True)
    num = torch.sum((benford_distribution - x_mean) * (data_distribution - y_mean), dim=1)
    den = torch.sqrt(
        torch.sum((benford_distribution - x_mean) ** 2, dim=1)
        * torch.sum((data_distribution - y_mean) ** 2, dim=1)
    )
    corr = torch.nan_to_num(num / den, nan=0.0)
    return corr.item() if corr.numel() == 1 else corr


def interquantile_range_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    return quantile_torch(array, 0.75, axis) - quantile_torch(array, 0.25, axis)


def energy_torch(array: torch.Tensor, axis=-1) -> float | torch.Tensor:
    axis = axis%array.ndim
    energy = torch.sum(array ** 2, dim=axis) / array.shape[axis]
    return energy.item() if energy.numel() == 1 else energy


def autocorrelation_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    axis = axis%x.ndim
    lagged = torch.roll(x, shifts=1, dims=axis)
    x_centered = x - x.mean(dim=axis, keepdim=True)
    lagged_centered = lagged - lagged.mean(dim=axis, keepdim=True)
    num = (x_centered * lagged_centered).sum(dim=axis)
    denom = torch.sqrt((x_centered ** 2).sum(dim=axis) * (lagged_centered ** 2).sum(dim=axis))
    corr = num / denom.clamp(min=1e-12)
    return corr.item() if corr.numel() == 1 else corr


def zero_crossing_rate_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    axis = axis%x.ndim
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
    axis = axis%x.ndim
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
    if axis != -1:
        x = x.transpose(axis, -1)
    if not torch.is_floating_point(x):
        x = x.float()
    T = x.shape[-1]
    span = max(int(T / 10), 2)
    alpha = 2 / (span + 1)
    weights = (1 - alpha) ** torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    ema = torch.sum(x * weights, dim=-1)
    return ema.item() if ema.numel() == 1 else ema


def mean_moving_median_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if axis != -1:
        x = x.transpose(axis, -1)
    T = x.shape[-1]
    span = max(int(T / 10), 2)
    span = min(span, T)
    medians = []
    for i in range(T - span + 1):
        window = x[..., i:i + span]
        medians.append(window.median(dim=-1).values)
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
    return mobility


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


def hurst_exponent_torch(x: torch.Tensor, axis: int = -1) -> float | torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    B, T = x.shape[-2:]
    t = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
    y = torch.cumsum(x, dim=axis)
    ave_t = y / t
    S_T = torch.zeros((B, T), device=x.device, dtype=x.dtype)
    R_T = torch.zeros((B, T), device=x.device, dtype=x.dtype)
    for i in range(T):
        S_T[:, i] = x[:, :i + 1].std(dim=axis, unbiased=False)
        X_T = y - t * ave_t[:, i].unsqueeze(1)
        R_T[:, i] = X_T[:, :i + 1].max(dim=axis).values - X_T[:, :i + 1].min(dim=axis).values
    rs = R_T / (S_T + 1e-12)
    rs = torch.log(rs[:, 1:])
    n = torch.log(t[1:])
    A = torch.stack([n, torch.ones_like(n)], dim=1)
    H = torch.empty(B, device=x.device, dtype=x.dtype)
    for b in range(B):
        sol = torch.linalg.lstsq(A, rs[b].unsqueeze(1)).solution.squeeze()
        m, c = sol[0], sol[1]
        H[b] = m
    return H.item() if H.numel() == 1 else H


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
    return  res.item() if res.numel() == 1 else res

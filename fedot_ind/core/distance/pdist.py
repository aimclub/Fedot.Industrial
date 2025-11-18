import torch
import torch.nn.functional as F


def torch_pdist(X: torch.Tensor, metric: str = 'euclidean', p: float = 3) -> float:
    """Pairwise distance
    """
    if metric == 'euclidean':
        XX = torch.sum(X * X, dim=1, keepdim=True)
        dist_sq = XX + XX.T - 2 * (X @ X.T)
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return torch.sqrt(dist_sq)
    elif metric == 'canberra':
        num = (X.unsqueeze(0) - X.unsqueeze(1)).abs()
        denom = X.abs().unsqueeze(0) + X.abs().unsqueeze(1) + 1e-8
        return torch.sum(num / denom, dim=-1)
    elif metric == 'cosine':
        Xn = F.normalize(X, p=2, dim=1)
        sim = torch.mm(Xn, Xn.T)
        return 1 - sim
    elif metric == 'cityblock':
        diff = (X.unsqueeze(0) - X.unsqueeze(1)).abs()
        return diff.sum(dim=-1)
    elif metric == 'correlation':
        Xc = X - X.mean(dim=1, keepdim=True)
        Xc = F.normalize(Xc, p=2, dim=1)
        sim = Xc @ Xc.T
        return 1 - sim
    elif metric == 'chebyshev':
        diff = (X.unsqueeze(0) - X.unsqueeze(1)).abs()
        return diff.max(dim=-1).values
    elif metric == 'minkowski':
        diff = (X.unsqueeze(0) - X.unsqueeze(1)).abs() ** p
        return diff.sum(dim=-1) ** (1.0 / p)
    else:
        return 0


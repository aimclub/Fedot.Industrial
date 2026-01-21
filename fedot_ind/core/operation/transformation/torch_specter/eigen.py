import torch


def aic_eigen_torch(s, N):
    s = torch.as_tensor(s, dtype=torch.float64)
    kaic = []
    n = s.numel()

    for k in range(n - 1):
        tail = s[k + 1 :]
        denom = n - k

        ak = torch.sum(tail) / denom
        gk = torch.exp(torch.sum(torch.log(tail)) / denom)

        val = (
            -2.0 * denom * N * torch.log(gk / ak)
            + 2.0 * k * (2.0 * n - k)
        )
        kaic.append(val)

    return torch.stack(kaic)


def mdl_eigen_torch(s, N):
    """
    Torch version of mdl_eigen
    """
    s = torch.as_tensor(s, dtype=torch.float64)

    kmdl = []
    n = s.numel()

    for k in range(0, n - 1):
        tail = s[k + 1 :]
        ak = torch.mean(tail)
        gk = torch.exp(torch.mean(torch.log(tail)))
        val = (
            -(n - k) * N * torch.log(gk / ak)
            + 0.5 * k * (2.0 * n - k) * torch.log(torch.tensor(float(N)))
        )
        kmdl.append(val)

    return torch.stack(kmdl)


def get_signal_space_torch(
    S,
    NP,
    NSIG=None,
    threshold=None,
    criteria="aic",
):
    """
    Determine number of signal components using
    AIC / MDL / threshold / manual NSIG
    """
    # Manual override
    if NSIG is not None:
        if NSIG <= 0:
            raise ValueError("NSIG must be positive")
        return NSIG

    # Threshold-based
    if threshold is not None:
        m = threshold * torch.min(S)
        NSIG = int((S > m).sum().item())
        if NSIG == 0:
            NSIG = 1
        return NSIG
    # Information criteria
    if criteria == "aic":
        crit = aic_eigen_torch(S, NP * 2)
    elif criteria == "mdl":
        crit = mdl_eigen_torch(S, NP * 2)
    else:
        raise ValueError("criteria must be 'aic' or 'mdl'")
    NSIG = int(torch.argmin(crit).item()) + 1

    return NSIG


def eigen_torch(
    x,
    P,
    NSIG=None,
    threshold=None,
    NFFT=4096,
    method="ev",
    criteria="aic",
):
    """
    Torch version of spectrum.eigen (EV / MUSIC)
    """
    x = torch.as_tensor(x, dtype=torch.complex128)
    N = x.numel()

    NP = N - P
    if 2 * NP <= P - 1:
        raise ValueError("Decrease P")
    NP = min(NP, 100)

    # Build FB matrix
    FB = torch.zeros((2 * NP, P), dtype=torch.complex128, device=x.device)

    for i in range(NP):
        for k in range(P):
            FB[i, k] = x[i - k + P - 1]
            FB[i + NP, k] = torch.conj(x[i + k + 1])

    # SVD
    U, S, Vh = torch.linalg.svd(FB, full_matrices=True)
    V = -Vh.conj().T

    # Determine signal subspace
    NSIG = get_signal_space_torch(
        S,
        2*NP,
        NSIG=NSIG,
        threshold=threshold,
        criteria=criteria,
    )

    PSD = torch.zeros(NFFT, dtype=torch.float64, device=x.device)

    for i in range(NSIG, P):
        Z = torch.zeros(NFFT, dtype=torch.complex128, device=x.device)
        Z[:P] = V[:P, i]

        Zf = torch.fft.fft(Z)

        if method == "music":
            PSD += Zf.abs() ** 2
        elif method == "ev":
            PSD += (Zf.abs() ** 2) / S[i]

    PSD = 1.0 / PSD

    # Rearrangement (same as spectrum)
    nby2 = NFFT // 2
    part1 = torch.flip(PSD[1:nby2 + 1], dims=[0])
    part2 = torch.flip(PSD[nby2:nby2*2], dims=[0])
    newpsd = torch.cat([part1, part2])

    return newpsd, S


def pev_torch(
    x,
    IP: int,
    NSIG=None,
    threshold=None,
    NFFT=4096,
    sampling=1.0,
    scale_by_freq=False,
):
    if NFFT is None:
        NFFT = x.shape[0]
    psd, eigenvalues = eigen_torch(
        x,
        IP,
        NSIG=NSIG,
        threshold=threshold,
        NFFT=NFFT,
        method="ev",
    )
    # One-sided PSD for real signal
    if not torch.is_complex(torch.as_tensor(x)):
        if NFFT % 2 == 0:
            psd = psd[: NFFT // 2 + 1] * 2
        else:
            psd = psd[: (NFFT + 1) // 2] * 2
        psd = torch.flip(psd, dims=[0])

    if scale_by_freq:
        df = sampling / NFFT
        psd = psd * (2 * torch.pi / df)

    return psd

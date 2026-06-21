from __future__ import annotations

import numpy as np


def compute_emd_diagnostics(values: np.ndarray, max_imfs: int = 5) -> dict:
    values = np.asarray(values, dtype=float)

    try:
        from PyEMD import EMD

        emd = EMD()
        imfs = emd.emd(values, max_imf=max_imfs)

        if len(imfs) == 0:
            return {
                "n_imfs": 0,
                "mode_energies": [],
                "dominant_mode": None,
            }

        energies = np.asarray([np.sum(imf ** 2) for imf in imfs], dtype=float)

        total = np.sum(energies)

        if total <= 0:
            mode_energies = np.zeros_like(energies)
        else:
            mode_energies = energies / total

        return {
            "n_imfs": int(len(imfs)),
            "mode_energies": mode_energies.tolist(),
            "dominant_mode": int(np.argmax(mode_energies)),
        }

    except Exception:
        return {
            "n_imfs": 0,
            "mode_energies": [],
            "dominant_mode": None,
        }


def compute_vmd_diagnostics(values: np.ndarray, k: int = 5) -> dict:
    values = np.asarray(values, dtype=float)

    try:
        from vmdpy import VMD

        alpha = 2000
        tau = 0
        dc = 0
        init = 1
        tol = 1e-7

        modes, _, _ = VMD(values, alpha, tau, k, dc, init, tol)

        if len(modes) == 0:
            return {
                "n_modes": 0,
                "mode_energies": [],
                "dominant_mode": None,
            }

        energies = np.asarray([np.sum(mode ** 2) for mode in modes], dtype=float)

        total = np.sum(energies)

        if total <= 0:
            mode_energies = np.zeros_like(energies)
        else:
            mode_energies = energies / total

        return {
            "n_modes": int(len(modes)),
            "mode_energies": mode_energies.tolist(),
            "dominant_mode": int(np.argmax(mode_energies)),
        }

    except Exception:
        return {
            "n_modes": 0,
            "mode_energies": [],
            "dominant_mode": None,
        }

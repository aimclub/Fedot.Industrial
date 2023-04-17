import numpy as np


def quantile_filter(input_data, predicted_data, threshold: float = 0.9, lp_norm: int = 1):
    reconstruction_error = np.linalg.norm(input_data - predicted_data, lp_norm, axis=1) / np.linalg.norm(
        input_data, lp_norm, axis=1)
    quantile = np.quantile(reconstruction_error, threshold)
    outlier_idx = [np.where(np.isclose(reconstruction_error, idx_outlier))[0][0]
                   for idx_outlier in reconstruction_error[reconstruction_error > quantile]]
    return outlier_idx

import torch

import torch
import torch.nn as nn


class DeepKernel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, base_kernel=None):
        """
        feature_extractor: нейросеть (nn.Module), переводящая x в латентное пространство.
        base_kernel: базовое ядро (например, RBFKernel). Если None, используется обычное скалярное произведение.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.base_kernel = base_kernel

    def _compute_batch_kernel(self, x, y):
        # x: (N_r, Q, 1, d) -> fx: (N_r, Q, 1, m)
        # y: (1, 1, N_l, d) -> fy: (1, 1, N_l, m)

        # fx = self.feature_extractor(x.float())
        # fy = self.feature_extractor(y.float())
        print(f"input dtype: {x.dtype}, feature extractor dtype: {next(self.feature_extractor.parameters()).dtype}")
        fx = self.feature_extractor(x)
        fy = self.feature_extractor(y)

        if self.base_kernel is not None:
            return self.base_kernel._compute_batch_kernel(fx, fy)
        else:
            return torch.sum(fx * fy, dim=-1)

    def _compute_single_kernel(self, x, y):
        # fx = self.feature_extractor(torch.as_tensor(x, dtype=torch.float32))
        # fy = self.feature_extractor(torch.as_tensor(y, dtype=torch.float32))
        print(f"input dtype: {x.dtype}, feature extractor dtype: {next(self.feature_extractor.parameters()).dtype}")
        fx = self.feature_extractor(x)
        fy = self.feature_extractor(y)

        if self.base_kernel is not None:
            return self.base_kernel._compute_single_kernel(fx, fy)
        return torch.dot(fx.view(-1), fy.view(-1)).item()

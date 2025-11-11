import numpy as np
import torch
import time

from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.data.kernel_matrix import TSTransformer
from fedot_ind.core.operation.transformation.data.kernel_matrix import TorchTSTransformer


def compare_hankel_and_transformer(ts, window_size=50, stride=1, rec_metric='cosine'):
    #Hankel
    hankel_np = HankelMatrix(time_series=ts, window_size=window_size, strides=stride)
    hankel_torch = HankelMatrix(time_series=torch.tensor(ts, dtype=torch.float32), 
                                window_size=window_size, strides=stride)
    traj_np = hankel_np.trajectory_matrix
    traj_torch = hankel_torch.trajectory_matrix
    if isinstance(traj_torch, torch.Tensor):
        traj_torch = traj_torch.cpu().numpy()
    print(f"Hankel shapes: numpy={traj_np.shape}, torch={traj_torch.shape}")
    hankel_diff = np.mean(np.abs(traj_np - traj_torch))
    print(f"Mean abs diff (Hankel): {hankel_diff:.6f}")


    #Recurrence
    ts_transformer_np = TSTransformer(time_series=traj_np, rec_metric=rec_metric)
    ts_transformer_torch = TorchTSTransformer(time_series=torch.tensor(traj_torch, dtype=torch.float32),
                                              rec_metric=rec_metric)

    start_np = time.time()
    rec_np = ts_transformer_np.ts_to_recurrence_matrix()
    t_np = time.time() - start_np
    # print(rec_np)
    start_torch = time.time()
    rec_torch = ts_transformer_torch.ts_to_recurrence_matrix()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_torch = time.time() - start_torch
    # print(rec_torch)
    if isinstance(rec_torch, torch.Tensor):
        rec_torch = rec_torch.cpu().numpy()
    
    diff = np.mean(np.abs(rec_np - rec_torch))
    print(f"Mean abs diff (Recurrence): {diff:.6f}")
    print(f"Speed: Numpy={t_np:.4f}s | Torch={t_torch:.4f}s | Speed-up={t_np / t_torch:.2f}x")
    return hankel_diff, diff, t_np, t_torch


if __name__ == "__main__":
    t = np.linspace(0, 10 * np.pi, 2000)
    ts = np.sin(t)
    hankel_diff, rec_diff, t_np, t_torch = compare_hankel_and_transformer(ts, rec_metric='canberra')

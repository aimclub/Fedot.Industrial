"""
Runtime for training deep OKHS encoder kernels.

Provides functions to fit DeepFDMDAutoencoder for use as pretrained kernel
in OKHS forecasting models, implementing Decoupled Spectral Training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm.auto import tqdm
    
    from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fdmd_net import DeepFDMDAutoencoder
    from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fractional_loss import DeepFractionalDMDLoss
    from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.time_grid_manager import TimeGridManager
except Exception:
    torch = None
    nn = None
    optim = None
    tqdm = None
    DeepFDMDAutoencoder = None
    DeepFractionalDMDLoss = None
    TimeGridManager = None


@dataclass
class DeepEncoderTrainingResult:
    """Result of encoder training."""
    
    encoder_state_dict: dict[str, Any]
    training_history: dict[str, list[float]]
    final_loss: float
    best_loss: float
    encoder_latent_dim: int
    config: DeepEncoderTrainingConfig
    metadata: dict[str, Any]


@dataclass
class DeepEncoderTrainingConfig:
    """Configuration for training a deep OKHS encoder."""
    latent_dim: int = 16
    hidden_layers: list[int] = None
    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 32
    device: str = 'cpu'
    reconstruction_weight: float = 1.0
    alpha_adjoint: float = 1.0  
    q: float = 0.7              
    dt: float = 1.0             
    n_quad_points: int = 20     
    patience: int = 10
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64]


def train_deep_encoder(
        trajectories: torch.Tensor | np.ndarray,
        config: DeepEncoderTrainingConfig | None = None,
) -> DeepEncoderTrainingResult:
    """
    Train a DeepFDMDAutoencoder on a set of phase space trajectories.
    Expects trajectories of shape [N_samples, Sequence_length, Dim].
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for deep encoder training")
    
    if config is None:
        config = DeepEncoderTrainingConfig()
        
    device = torch.device(config.device)
    dtype = torch.float64
    
    if isinstance(trajectories, np.ndarray):
        X_all = torch.tensor(trajectories, dtype=dtype, device=device)
    else:
        X_all = trajectories.to(dtype=dtype, device=device)
        
    if X_all.ndim == 2:
        X_all = X_all.unsqueeze(-1)
        
    N, S, d = X_all.shape
    
    time_manager = TimeGridManager(dt=config.dt)
    traj_list = [X_all[i] for i in range(N)]
    time_manager.fit(traj_list)
    
    t_grids_norm = torch.stack(time_manager.train_t_grids_norm_).to(device)
    T_norm_tensor = torch.full((N,), t_grids_norm[0, -1].item(), dtype=dtype, device=device)
    
    encoder = DeepFDMDAutoencoder(
        input_dim=d,
        latent_dim=config.latent_dim,
        hidden_layers=config.hidden_layers,
        dtype=dtype,
    ).to(device)
    
    adjoint_loss_fn = DeepFractionalDMDLoss(
        latent_dim=config.latent_dim, 
        q=config.q, 
        n_quad_points=config.n_quad_points, 
        device=device
    )
    recon_loss_fn = nn.MSELoss()
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(adjoint_loss_fn.parameters()), 
        lr=config.learning_rate,
        weight_decay=1e-5
    )


    with torch.no_grad():
        t_k_all, tau_nodes_all = adjoint_loss_fn.get_collocation_nodes(T_norm_tensor)
        X_tk_all = time_manager.interpolate_batch(X_all, t_grids_norm, t_k_all)          # (N, S, d)
        X_nodes_all = time_manager.interpolate_batch(X_all, t_grids_norm, tau_nodes_all) # (N, S, Q, d)
        X_start_all = X_all[:, 0, :]                                                     # (N, d)

    history = {
        'loss': [],
        'batch_losses': [],
        'reconstruction_loss': [],
        'adjoint_loss': []
    }
    
    best_loss = float('inf')
    patience_counter = 0
    indices = np.arange(N)
    
    encoder.train()
    pbar = tqdm(range(config.epochs), desc="Training encoder", disable=tqdm is None)
    
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        np.random.shuffle(indices)
        
        for start_idx in range(0, N, config.batch_size):
            batch_idx = indices[start_idx:start_idx + config.batch_size]
            
            x_batch = X_all[batch_idx]
            x_start_batch = X_start_all[batch_idx]
            x_tk_batch = X_tk_all[batch_idx]
            x_nodes_batch = X_nodes_all[batch_idx]
            t_k_batch = t_k_all[batch_idx]
            
            optimizer.zero_grad()
            
            _, x_recon = encoder(x_batch)
            loss_recon = recon_loss_fn(x_recon, x_batch)
            
            z_start = encoder.encode_trajectory(x_start_batch)
            z_tk = encoder.encode_trajectory(x_tk_batch)
            z_nodes = encoder.encode_trajectory(x_nodes_batch)
            
            loss_adj = adjoint_loss_fn(z_start, z_tk, z_nodes, t_k_batch)
            
            loss = config.reconstruction_weight * loss_recon + config.alpha_adjoint * loss_adj
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            history['reconstruction_loss'].append(loss_recon.item())
            history['adjoint_loss'].append(loss_adj.item())
            history['batch_losses'].append(loss.item())
            n_batches += 1
            
        avg_loss = epoch_loss / max(1, n_batches)
        history['loss'].append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in encoder.state_dict().items()}
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            if tqdm is not None:
                pbar.close()
            break
            
        pbar.update(1)
        if tqdm is not None:
            pbar.set_postfix({'loss': f"{avg_loss:.4e}", 'best': f"{best_loss:.4e}"})
            
    encoder.load_state_dict(best_state)
    encoder.eval()
    
    return DeepEncoderTrainingResult(
        encoder_state_dict=encoder.state_dict(),
        training_history=history,
        final_loss=avg_loss,
        best_loss=best_loss,
        encoder_latent_dim=config.latent_dim,
        config=config,
        metadata={
            'dataset_size': N,
            'epochs_trained': epoch + 1,
            'early_stopped': patience_counter >= config.patience,
        },
    )


def save_trained_encoder(
        result: DeepEncoderTrainingResult,
        save_path: str | Any,
) -> None:
    """Save trained encoder state dict to file."""
    import torch
    from pathlib import Path
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.encoder_state_dict, save_path)


def load_trained_encoder(
        load_path: str | Any,
        latent_dim: int,
        input_dim: int = 1,
) -> Any:
    """Load trained encoder from file."""
    import torch
    from pathlib import Path
    
    try:
        from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fdmd_net import DeepFDMDAutoencoder
    except ImportError:
        raise RuntimeError("DeepFDMDAutoencoder is not available in current environment.")
        
    load_path = Path(load_path)
    
    encoder = DeepFDMDAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        dtype=torch.float64
    )
    state_dict = torch.load(load_path, map_location='cpu')
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    return encoder
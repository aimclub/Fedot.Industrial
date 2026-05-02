"""
Runtime for training deep OKHS encoder kernels.

Provides functions to fit DeepFDMDAutoencoder for use as pretrained kernel
in OKHS forecasting models.
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
except Exception:
    torch = None
    nn = None
    optim = None
    tqdm = None


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
    regularization_weight: float = 0.1
    patience: int = 10
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64]


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


def _build_trajectory_batches(
        time_series: np.ndarray,
        window_size: int,
        batch_size: int,
) -> list[np.ndarray]:
    """Create batches of trajectory windows from time series."""
    series = np.asarray(time_series, dtype=float).reshape(-1)
    trajectories = [
        series[i:i + window_size]
        for i in range(max(0, len(series) - window_size))
    ]
    
    if not trajectories:
        return []
    
    # Shuffle and batch
    shuffled = np.random.permutation(len(trajectories))
    batches = []
    
    for start_idx in range(0, len(shuffled), batch_size):
        end_idx = min(start_idx + batch_size, len(shuffled))
        batch_indices = shuffled[start_idx:end_idx]
        batch = np.array([trajectories[i] for i in batch_indices], dtype=float)
        batches.append(batch)
    
    return batches


def train_deep_encoder(
        time_series: np.ndarray,
        config: DeepEncoderTrainingConfig | None = None,
) -> DeepEncoderTrainingResult:
    """
    Train a DeepFDMDAutoencoder on a time series.
    
    Parameters
    ----------
    time_series : np.ndarray
        1D array of time series values
    config : DeepEncoderTrainingConfig, optional
        Training configuration
    
    Returns
    -------
    DeepEncoderTrainingResult
        Trained encoder state dict and training history
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for deep encoder training")
    
    if config is None:
        config = DeepEncoderTrainingConfig()
    
    from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fdmd_net import (
        DeepFDMDAutoencoder,
    )
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    series = np.asarray(time_series, dtype=float).reshape(-1)
    
    # Determine input dimension (window size or 1 for univariate)
    window_size = max(4, min(20, len(series) // 4))
    input_dim = 1  # For univariate forecasting
    
    # Create batches
    batches = _build_trajectory_batches(series, window_size, config.batch_size)
    if not batches:
        raise ValueError(f"Cannot create batches from series of length {len(series)}")
    
<<<<<<< Updated upstream
=======
    # Initialize model
>>>>>>> Stashed changes
    device = torch.device(config.device)
    encoder = DeepFDMDAutoencoder(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        hidden_layers=config.hidden_layers,
        dtype=torch.float32,
    )
    encoder.to(device)
    encoder.train()
    
<<<<<<< Updated upstream
    optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    
=======
    # Optimizer
    optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    
    # Training history
>>>>>>> Stashed changes
    history = {
        'loss': [],
        'batch_losses': [],
        'reconstruction_loss': [],
    }
    
    best_loss = float('inf')
    patience_counter = 0
    
<<<<<<< Updated upstream
=======
    # Training loop
>>>>>>> Stashed changes
    pbar = tqdm(range(config.epochs), desc="Training encoder", disable=tqdm is None)
    
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in batches:
<<<<<<< Updated upstream
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            if batch_tensor.ndim == 2:
                batch_tensor = batch_tensor.unsqueeze(-1)
            
            optimizer.zero_grad()
            z, x_recon = encoder(batch_tensor)
            
            recon_loss = nn.MSELoss()(x_recon, batch_tensor)
            
            latent_regularization = config.regularization_weight * torch.mean(torch.norm(z, dim=-1) ** 2)
            
            loss = config.reconstruction_weight * recon_loss + latent_regularization
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

=======
            # Convert batch to tensor
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            # Reshape for network: (batch_size, seq_len) -> (batch_size, seq_len, input_dim)
            if batch_tensor.ndim == 2:
                batch_tensor = batch_tensor.unsqueeze(-1)
            
            # Forward pass
            optimizer.zero_grad()
            z, x_recon = encoder(batch_tensor)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(x_recon, batch_tensor)
            
            # Regularization: encourage latent space to stay compact
            latent_regularization = config.regularization_weight * torch.mean(torch.norm(z, dim=-1) ** 2)
            
            # Total loss
            loss = config.reconstruction_weight * recon_loss + latent_regularization
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
>>>>>>> Stashed changes
            epoch_loss += loss.item()
            history['reconstruction_loss'].append(recon_loss.item())
            history['batch_losses'].append(loss.item())
            n_batches += 1
        
        avg_loss = epoch_loss / max(1, n_batches)
        history['loss'].append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            if tqdm is not None:
                pbar.close()
            break
        
        pbar.update(1)
        if tqdm is not None:
            pbar.set_postfix({'loss': avg_loss, 'best': best_loss})
    
    encoder.eval()
    
    return DeepEncoderTrainingResult(
        encoder_state_dict=encoder.state_dict(),
        training_history=history,
        final_loss=avg_loss,
        best_loss=best_loss,
        encoder_latent_dim=config.latent_dim,
        config=config,
        metadata={
            'series_length': len(series),
            'window_size': window_size,
            'n_batches': n_batches,
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
    from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fdmd_net import (
        DeepFDMDAutoencoder,
    )
    
    load_path = Path(load_path)
    
    encoder = DeepFDMDAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
    )
    state_dict = torch.load(load_path, map_location='cpu')
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    return encoder

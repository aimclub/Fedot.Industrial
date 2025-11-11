import torch
import numpy as np
import time
from sklearn.decomposition import PCA
from fedot_ind.core.operation.transformation.torch_backend.tabular.tabular_extractor import PCA_transformation



n_samples, n_features = 30000, 10000
X_np = np.random.randn(n_samples, n_features).astype(np.float32)
X_torch = torch.tensor(X_np)
device = 'cpu'
start = time.time()
sk_pca = PCA(0.975)
X_sk_reduced = sk_pca.fit_transform(X_np)
X_sk_reconstructed = sk_pca.inverse_transform(X_sk_reduced)
sk_time = time.time() - start
sk_mse = np.mean((X_np - X_sk_reconstructed) ** 2)
print(f"SKLEARN PCA:  time={sk_time:.4f}s, mse={sk_mse:.6f}")
dev = ['cpu', 'cuda']
for d in dev:
    if d == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'  
        else:
            print('cuda error')
            break
        X_torch = X_torch.to(device)
    start = time.time()
    torch_pca = PCA_transformation(explained_variance=0.97)
    torch_pca.fit(X_torch)
    X_torch_reduced = torch_pca(X_torch)
    X_torch_reconstructed = X_torch_reduced @ torch_pca.components + X_torch.mean()
    torch_time = time.time() - start
    torch_mse = torch.mean((X_torch - X_torch_reconstructed) ** 2).item()
    print(f"TORCH PCA ({device}):     time={torch_time:.4f}s, mse={torch_mse:.6f}")
    print(f"Speed ratio (torch/sklearn): {torch_time / sk_time:.2f}x", "\n")

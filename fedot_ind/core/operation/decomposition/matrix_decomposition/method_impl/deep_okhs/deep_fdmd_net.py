import torch
import torch.nn as nn
from typing import List, Tuple


class DeepFDMDAutoencoder(nn.Module):
    """
    Автоэнкодер для обучения нелинейного латентного пространства \tilde{H} в методе Deep OKHS.
    Состоит из энкодера и декодера, обучающихся совместно с оператором Лиувилля W.
    Имеет пропускные связи для облегчения оптимизации и обеспечения начальной идентичности.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_layers: List[int] = [64, 64], dtype=torch.float64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dtype = dtype

        self.encoder_net = self._build_mlp(input_dim, latent_dim, hidden_layers)
        self.decoder_net = self._build_mlp(latent_dim, input_dim, list(reversed(hidden_layers)))

        self.enc_skip = nn.Linear(input_dim, latent_dim, bias=False, dtype=dtype)
        self.dec_skip = nn.Linear(latent_dim, input_dim, bias=False, dtype=dtype)

        self._initialize_identity_mapping()

    def _build_mlp(self, in_features: int, out_features: int, hidden_dims: List[int]) -> nn.Sequential:
        """Построение MLP с заданными скрытыми слоями и активациями."""
        modules = []
        curr_in = in_features
        for h_dim in hidden_dims:
            modules.append(nn.Linear(curr_in, h_dim, dtype=self.dtype))
            modules.append(nn.ELU())
            curr_in = h_dim
        modules.append(nn.Linear(curr_in, out_features, dtype=self.dtype))
        return nn.Sequential(*modules)

    def _initialize_identity_mapping(self):
        """
        Инициализация весов, чтобы на старте выполнялось:
        encoder(x) ≈ x (с дополнением нулями, если latent_dim > input_dim)
        decoder(z) ≈ z (с усечением, если input_dim < latent_dim)
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.encoder_net[-1].weight)
        nn.init.zeros_(self.encoder_net[-1].bias)
        nn.init.zeros_(self.decoder_net[-1].weight)
        nn.init.zeros_(self.decoder_net[-1].bias)

        # Для энкодера: R^d -> R^m
        enc_eye = torch.eye(self.latent_dim, self.input_dim, dtype=self.dtype)
        with torch.no_grad():
            self.enc_skip.weight.copy_(enc_eye)

        # Для декодера: R^m -> R^d
        dec_eye = torch.eye(self.input_dim, self.latent_dim, dtype=self.dtype)
        with torch.no_grad():
            self.dec_skip.weight.copy_(dec_eye)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Композиция: нелинейный путь + линейная "тождественная" проекция
        z = self.encoder_net(x) + self.enc_skip(x)
        x_recon = self.decoder_net(z) + self.dec_skip(z)

        return z, x_recon

    def encode_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(x) + self.enc_skip(x)


class AdjointBasisEncoder(nn.Module):
    """
    Нейросеть для аппроксимации базиса инвариантного подпространства 
    сопряженного дробного оператора Лиувилля (A_{f, q}^*).
    """
    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int, 
                 hidden_layers: List[int] = [64, 64], 
                 activation: nn.Module = nn.ELU()):
        
        super(AdjointBasisEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation)
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

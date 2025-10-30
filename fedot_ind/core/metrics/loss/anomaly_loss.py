import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Контрастивная функция потерь для обучения представлений
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class FocalLoss(nn.Module):
    """
    Focal Loss для работы с несбалансированными данными
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class ReconstructionLoss(nn.Module):
    """
    Комбинированная функция потерь для реконструкции
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        # MSE loss
        mse_loss = F.mse_loss(reconstructed, original)

        # Cosine similarity loss
        cosine_loss = 1 - F.cosine_similarity(reconstructed, original, dim=-1).mean()

        # Combined loss
        combined_loss = self.alpha * mse_loss + self.beta * cosine_loss

        return combined_loss


class DynamicThresholdLoss(nn.Module):
    """
    Функция потерь с динамическим порогом для аномалий
    """

    def __init__(self, contamination: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.contamination = contamination
        self.margin = margin

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Динамический порог на основе quantile
        threshold = torch.quantile(scores, 1 - self.contamination)

        # Loss для нормальных точек (должны быть ниже порога)
        normal_mask = (targets == 0)
        normal_loss = F.relu(scores[normal_mask] - (threshold - self.margin))

        # Loss для аномальных точек (должны быть выше порога)
        anomaly_mask = (targets == 1)
        anomaly_loss = F.relu((threshold + self.margin) - scores[anomaly_mask])

        total_loss = torch.cat([normal_loss, anomaly_loss]).mean()

        return total_loss


class MultiScaleLoss(nn.Module):
    """
    Мультимасштабная функция потерь для временных рядов
    """

    def __init__(self, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        total_loss = 0

        for scale in self.scales:
            # Агрегация на разных масштабах
            if scale > 1:
                reconstructed_scaled = F.avg_pool1d(reconstructed.unsqueeze(1), scale).squeeze(1)
                original_scaled = F.avg_pool1d(original.unsqueeze(1), scale).squeeze(1)
            else:
                reconstructed_scaled = reconstructed
                original_scaled = original

            scale_loss = F.mse_loss(reconstructed_scaled, original_scaled)
            total_loss += scale_loss

        return total_loss / len(self.scales)

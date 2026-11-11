"""Compact tensor attention summaries with no domain registry imports."""
import torch


def summarize_bottleneck_attn(
    attn_list: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compact cross/self-attention diagnostics over all bottleneck layers."""

    if not attn_list:
        return {}

    cross_means: list[float] = []
    cross_stds: list[float] = []
    self_means: list[float] = []
    self_stds: list[float] = []

    for layer_attn in attn_list:
        cross = layer_attn.get("cross_attn")
        self_attn = layer_attn.get("self_attn")
        if cross is not None:
            detached = cross.detach()
            cross_means.append(float(detached.mean().item()))
            cross_stds.append(float(detached.std(unbiased=False).item()))
        if self_attn is not None:
            detached = self_attn.detach()
            self_means.append(float(detached.mean().item()))
            self_stds.append(float(detached.std(unbiased=False).item()))

    summary: dict[str, float] = {"num_layers_with_attn": float(len(attn_list))}
    if cross_means:
        summary["cross_attn_mean"] = sum(cross_means) / len(cross_means)
        summary["cross_attn_std"] = sum(cross_stds) / len(cross_stds)
    if self_means:
        summary["self_attn_mean"] = sum(self_means) / len(self_means)
        summary["self_attn_std"] = sum(self_stds) / len(self_stds)
    return summary

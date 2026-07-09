"""Shared validation and normalization rules for NN model configs."""

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from fedot_ind.core.multimodal.enums import MultimodalModality

if TYPE_CHECKING:
    from fedot_ind.core.models.nn.network_impl.encoders.config import EncoderConfig


class EncoderFamily(str, Enum):
    """Supported encoder families."""

    cnn = "cnn"
    mlp = "mlp"


def normalize_encoder_family(value: EncoderFamily | str) -> EncoderFamily:
    """Convert user-provided family value to EncoderFamily."""

    if isinstance(value, EncoderFamily):
        return value
    try:
        return EncoderFamily(str(value))
    except ValueError as exc:
        supported = [family.value for family in EncoderFamily]
        raise ValueError(
            f"Unsupported encoder family {value!r}. Supported values: {supported}."
        ) from exc


def normalize_modality(modality: MultimodalModality | str) -> MultimodalModality:
    """Convert user-provided modality value to MultimodalModality."""

    if isinstance(modality, MultimodalModality):
        return modality
    try:
        return MultimodalModality(str(modality))
    except ValueError as exc:
        supported = [item.value for item in MultimodalModality]
        raise ValueError(
            f"Unsupported modality {modality!r}. Supported values: {supported}."
        ) from exc


def validate_conv_block_config(config: "ConvBlockConfig") -> None:
    """Validate conv block configuration."""

    if config.out_channels <= 0:
        raise ValueError("Conv block out_channels must be positive.")
    if config.kernel_size <= 0:
        raise ValueError("Conv block kernel_size must be positive.")
    if config.stride <= 0:
        raise ValueError("Conv block stride must be positive.")
    if config.pool_kernel_size <= 0:
        raise ValueError("Conv block pool_kernel_size must be positive.")
    if config.pool_stride <= 0:
        raise ValueError("Conv block pool_stride must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("Conv block dropout must be in [0.0, 1.0).")

def validate_encoder_config(config: "EncoderConfig") -> None:
    """Validate shared and family-specific encoder config invariants."""

    if config.d_model <= 0:
        raise ValueError("d_model must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("Encoder dropout must be in [0.0, 1.0).")

    if config.family is EncoderFamily.cnn:
        _validate_cnn_encoder_config(config)
        return
    if config.family is EncoderFamily.mlp:
        _validate_mlp_encoder_config(config)
        return
    raise ValueError(f"Unsupported encoder family: {config.family}.")


def _validate_cnn_encoder_config(config: "EncoderConfig") -> None:
    if config.input_rank not in (3, 4):
        raise ValueError("CNN encoder input_rank must be 3 (1D) or 4 (2D).")
    if config.in_channels is None or config.in_channels <= 0:
        raise ValueError("CNN encoder requires positive in_channels.")
    if not config.conv_blocks:
        raise ValueError("CNN encoder requires at least one conv block.")
    if config.hidden_dims:
        raise ValueError("CNN encoder must not define hidden_dims.")
    if config.in_features is not None:
        raise ValueError("CNN encoder must not define in_features.")


def _validate_mlp_encoder_config(config: "EncoderConfig") -> None:
    if config.input_rank != 2:
        raise ValueError("MLP encoder input_rank must be 2.")
    if config.in_features is None or config.in_features <= 0:
        raise ValueError("MLP encoder requires positive in_features.")
    if not config.hidden_dims:
        raise ValueError("MLP encoder requires non-empty hidden_dims.")
    for hidden_dim in config.hidden_dims:
        if hidden_dim <= 0:
            raise ValueError("MLP hidden_dims values must be positive.")
    if config.conv_blocks:
        raise ValueError("MLP encoder must not define conv_blocks.")
    if config.in_channels is not None:
        raise ValueError("MLP encoder must not define in_channels.")


def build_encoder_config_map(
    entries: Sequence[tuple[MultimodalModality | str, Any]],
) -> dict[MultimodalModality, Any]:
    """Build a modality -> encoder config mapping with duplicate checks."""

    normalized: dict[MultimodalModality, Any] = {}
    for raw_modality, config in entries:
        modality = normalize_modality(raw_modality)
        if modality in normalized:
            raise ValueError(
                f"Duplicate modality definition in encoder config: {modality.value}."
            )
        normalized[modality] = config
    if not normalized:
        raise ValueError("Encoder configuration map must contain at least one modality.")
    return normalized

from enum import Enum


class EncoderFamily(str, Enum):
    """Supported encoder families."""

    cnn = "cnn"
    mlp = "mlp"

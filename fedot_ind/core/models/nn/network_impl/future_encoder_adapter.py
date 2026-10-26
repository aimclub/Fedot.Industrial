"""Compatibility alias preserving legacy module-level registry patching."""

import sys

from fedot_ind.core.multimodal import encoders as _implementation

sys.modules[__name__] = _implementation

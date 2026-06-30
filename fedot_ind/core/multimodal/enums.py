from enum import Enum
from typing import Sequence


class MultimodalModality(Enum):
    raw = "raw"
    stats = "stats"
    gaf = "gaf"
    stft = "stft"
    mtf = "mtf"


class NormalizationMethod(str, Enum):
    imputation = "imputation"
    feature_standardization = "feature_standardization"
    image_standardization = "image_standardization"
    log1p = "log1p"


NormalizationStep = NormalizationMethod
NormalizationConfig = dict[MultimodalModality, Sequence[NormalizationStep]]

from enum import Enum


class MultimodalModality(Enum):
    raw = "raw"
    stats = "stats"
    gaf = "gaf"
    stft = "stft"
    mtf = "mtf"

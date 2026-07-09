from typing import Any
from fedot_ind.core.operation.transformation.torch_backend.enums import (
    StatisticalFeature,
    STAT_FEATURE_CONFIG,
)


def normalize_feature_key(key: Any) -> StatisticalFeature:
    if isinstance(key, StatisticalFeature):
        return key
    if hasattr(key, "value"):
        key = key.value
    return StatisticalFeature(str(key))


def build_default_feature_config(methods: dict) -> STAT_FEATURE_CONFIG:
    config: STAT_FEATURE_CONFIG = {}
    for raw_key in methods:
        feature = normalize_feature_key(raw_key)
        config[feature] = {}
    return config


def method_registry(methods: dict) -> dict[StatisticalFeature, callable]:
    registry: dict[StatisticalFeature, callable] = {}
    for raw_key, method in methods.items():
        registry[normalize_feature_key(raw_key)] = method
    return registry
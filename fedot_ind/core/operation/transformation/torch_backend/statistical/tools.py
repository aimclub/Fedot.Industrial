from collections.abc import Callable
from typing import Any

from fedot_ind.core.operation.transformation.torch_backend.enums import (
    STAT_FEATURE_CONFIG,
    StatisticalFeature,
)


def normalize_feature_key(key: Any) -> StatisticalFeature:
    if isinstance(key, StatisticalFeature):
        return key
    if hasattr(key, "value"):
        key = key.value
    key_value = str(key)
    if key_value.endswith("_"):
        key_value = key_value[:-1]
    return StatisticalFeature(key_value)


def build_default_feature_config(methods: dict) -> STAT_FEATURE_CONFIG:
    config: STAT_FEATURE_CONFIG = {}
    for raw_key in methods:
        feature = normalize_feature_key(raw_key)
        config[feature] = {}
    return config


def method_registry(methods: dict) -> dict[StatisticalFeature, Callable]:
    registry: dict[StatisticalFeature, Callable] = {}
    for raw_key, method in methods.items():
        registry[normalize_feature_key(raw_key)] = method
    return registry

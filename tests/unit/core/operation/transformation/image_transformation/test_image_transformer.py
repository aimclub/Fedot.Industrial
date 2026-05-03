import numpy as np
import pytest

from fedot_ind.core.operation.transformation.torch_backend.image_transformation.image_transformer import (
    ImageTransformer,
)
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.types import (
    ImageTransformationType,
)


@pytest.fixture
def data_2d() -> np.ndarray:
    return np.random.rand(100, 100)


@pytest.fixture
def data_3d() -> np.ndarray:
    return np.random.rand(100, 100, 100)


def test_2d_data(data_2d):
    transformer = ImageTransformer(
        params={
            "transformation_type": ImageTransformationType.MTF,
            "transfromation_params": {
                "image_size": 1.0,
                "n_bins": 8,
                "strategy": "quantile",
                "overlapping": False,
                "flatten": False,
            },
        }
    )
    result = transformer.transform(data_2d)
    assert result.shape == (100, 100, 100)


def test_3d_data(data_3d):
    transformer = ImageTransformer(
        params={
            "transformation_type": ImageTransformationType.MTF,
            "transfromation_params": {
                "image_size": 1.0,
                "n_bins": 8,
                "strategy": "quantile",
                "overlapping": False,
                "flatten": False,
            },
        }
    )
    result = transformer.transform(data_3d)
    assert result.shape == (10000, 100, 100)

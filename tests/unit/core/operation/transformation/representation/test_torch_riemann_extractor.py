"""Integration checks for the Torch-backed Riemann feature extractor."""

import numpy as np
import pytest
import torch
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from pyriemann.estimation import BlockCovariances, Covariances, Shrinkage
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils import mean_covariance
from pyriemann.utils.distance import distance

from fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding import (
    RiemannExtractor,
)


def _view(builder: str = "covariance", **params) -> dict:
    """Build a canonical single-view configuration for integration tests."""
    return {
        "name": builder,
        "builder": builder,
        "weight": 1.0,
        "shrinkage": 0.1,
        "params": params,
    }


def _input_data(features: np.ndarray, target: np.ndarray | None = None) -> InputData:
    """Build minimal FEDOT input data for extractor integration tests."""
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


@pytest.fixture
def multichannel_data() -> InputData:
    """Return deterministic multichannel signals and binary labels."""
    generator = np.random.default_rng(42)
    return _input_data(generator.normal(size=(8, 4, 192)), np.repeat([0, 1], 4))


def test_torch_extractor_tangent_features_match_pyriemann_covariance(multichannel_data):
    """Match the legacy dense PyRiemann tangent pipeline for an AIRM mean."""
    extractor = RiemannExtractor(
        {"views": [_view(estimator="scm")], "feature_mode": "tangent", "tangent_metric": "riemann"}
    )

    actual = extractor.fit(multichannel_data)._transform(multichannel_data)
    matrices = Shrinkage().fit_transform(Covariances(estimator="scm").fit_transform(multichannel_data.features))
    expected = TangentSpace(metric="riemann").fit_transform(matrices)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_torch_extractor_tangent_features_match_compact_pyriemann_blocks(multichannel_data):
    """Return the useful block tangent coordinates without dense structural zeros."""
    extractor = RiemannExtractor(
        {
            "views": [_view("grouped_covariance", group_sizes=[1, 3], estimator="scm")],
            "feature_mode": "tangent",
            "tangent_metric": "riemann",
        }
    )

    actual = extractor.fit(multichannel_data)._transform(multichannel_data)
    matrices = Shrinkage().fit_transform(
        BlockCovariances(estimator="scm", block_size=[1, 3]).fit_transform(multichannel_data.features)
    )
    legacy = TangentSpace(metric="riemann").fit_transform(matrices)
    expected = legacy[:, [0, 4, 5, 6, 7, 8, 9]] / np.sqrt(2.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_torch_extractor_mdm_features_match_pyriemann(multichannel_data):
    """Match class-wise Log-Euclidean MDM distances from PyRiemann."""
    extractor = RiemannExtractor(
        {
            "views": [_view(estimator="scm")],
            "feature_mode": "mdm",
            "mdm_centroid_scope": "class",
            "mdm_metric": "logeuclid",
        }
    )

    actual = extractor.fit(multichannel_data)._transform(multichannel_data)
    matrices = Shrinkage().fit_transform(Covariances(estimator="scm").fit_transform(multichannel_data.features))
    centroids = [mean_covariance(matrices[multichannel_data.target == label], metric="logeuclid") for label in [0, 1]]
    expected = np.column_stack([distance(matrices, centroid, metric="logeuclid") for centroid in centroids])

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_torch_extractor_uses_median_centroid_for_tangent_features(multichannel_data):
    """Wire the Torch geometric median into the tangent projection path."""
    extractor = RiemannExtractor(
        {
            "views": [_view(estimator="scm")],
            "feature_mode": "tangent",
            "centroid_type": "median",
            "tangent_metric": "logeuclid",
        }
    )

    features = extractor.fit(multichannel_data)._transform(multichannel_data)

    assert extractor.centroid_type == "median"
    assert features.shape == (multichannel_data.features.shape[0], 10)
    assert np.all(np.isfinite(features))


@pytest.mark.parametrize("representation_type", ["covariance", "block", "cospectra"])
def test_torch_extractor_propagates_configured_float32_dtype(multichannel_data, representation_type):
    """Use ``torch_dtype`` consistently for every SPD representation path."""
    builder = {
        "covariance": "covariance",
        "block": "grouped_covariance",
        "cospectra": "cospectra",
    }[representation_type]
    builder_params = {"estimator": "scm"}
    if representation_type == "block":
        builder_params = {"group_sizes": [2, 2], "estimator": "scm"}
    if representation_type == "cospectra":
        builder_params = {"fmin": 1.0, "fmax": 32.0, "fs": 100.0}
    params = {
        "views": [_view(builder, **builder_params)],
        "feature_mode": "tangent",
        "torch_dtype": "float32",
    }

    extractor = RiemannExtractor(params).fit(multichannel_data)
    features = extractor._transform(multichannel_data)

    centroid = extractor.tangent_product_centroid_[builder]
    assert centroid.to_dense().dtype is torch.float32
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


@pytest.mark.parametrize("metric", ["logdet", "kullback", "wasserstein"])
def test_torch_extractor_rejects_unported_metrics(metric):
    """Fail explicitly instead of silently dispatching an unsupported metric."""
    with pytest.raises(NotImplementedError, match="not implemented for the Torch backend"):
        RiemannExtractor({"views": [_view()], "mdm_metric": metric})

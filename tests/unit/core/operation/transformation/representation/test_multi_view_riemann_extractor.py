"""Product-manifold integration tests for the multi-view Riemann extractor."""

import math

import numpy as np
import pytest
import torch
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding import (
    RiemannExtractor,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_mdm import (
    TorchClassCentroids,
    TorchMDMDistances,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    BaseTorchSPDBuilder,
    RaggedBlockSPDBatch,
    TorchBlockCovariances,
    TorchCovariances,
    TorchShrinkage,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_tangent_space import (
    TorchSPDCentroid,
    TorchTangentSpace,
)


def _input_data(
    features: np.ndarray,
    target: np.ndarray | None = None,
    task_type: TaskTypesEnum = TaskTypesEnum.classification,
) -> InputData:
    """Build FEDOT input data for direct extractor tests."""
    return InputData(
        idx=np.arange(features.shape[0]),
        features=features,
        target=target,
        task=Task(task_type),
        data_type=DataTypesEnum.image,
    )


@pytest.fixture
def multichannel_data() -> InputData:
    """Return deterministic four-channel signals with two balanced classes."""
    generator = np.random.default_rng(42)
    features = generator.normal(size=(8, 4, 192))
    return _input_data(features, np.repeat([0, 1], 4))


def _raw_view(weight: float = 1.0) -> dict:
    """Return the canonical raw-covariance view configuration."""
    return {
        "name": "raw",
        "builder": "covariance",
        "weight": weight,
        "shrinkage": 0.1,
        "params": {"estimator": "scm", "estimator_params": {}},
    }


def _grouped_view(weight: float = 1.0) -> dict:
    """Return a two-block grouped-covariance view configuration."""
    return {
        "name": "groups",
        "builder": "grouped_covariance",
        "weight": weight,
        "shrinkage": 0.1,
        "params": {
            "group_sizes": [2, 2],
            "estimator": "scm",
            "estimator_params": {},
        },
    }


def _manual_views(features: np.ndarray):
    """Build the raw and grouped SPD views independently of the extractor."""
    signals = torch.from_numpy(features)
    raw = TorchShrinkage(0.1).fit_transform(
        TorchCovariances(estimator="scm").fit_transform_spd(signals)
    )
    grouped = TorchShrinkage(0.1).fit_transform(
        TorchBlockCovariances(
            block_size=[2, 2], estimator="scm"
        ).fit_transform_spd(signals)
    )
    return raw, grouped


def test_multi_view_tangent_features_match_weighted_direct_sum(multichannel_data):
    """Tangent output must be the weighted direct sum of component coordinates."""
    extractor = RiemannExtractor({
        "views": [_raw_view(weight=2.0), _grouped_view(weight=3.0)],
        "feature_mode": "tangent",
        "tangent_metric": "riemann",
    })

    actual = extractor.fit_transform(multichannel_data).predict
    raw, grouped = _manual_views(multichannel_data.features)
    raw_reference = TorchSPDCentroid(metric="riemann").fit(raw).centroid_
    grouped_reference = TorchSPDCentroid(metric="riemann").fit(grouped).centroid_
    expected = torch.cat([
        math.sqrt(2.0) * TorchTangentSpace(raw_reference, metric="riemann").transform(raw),
        math.sqrt(3.0 / 2.0)
        * TorchTangentSpace(grouped_reference, metric="riemann").transform(grouped),
    ], dim=1)

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-7, atol=1e-8)


def test_multi_view_class_mdm_matches_weighted_product_distance(multichannel_data):
    """Class MDM must expose one aggregated product distance per class."""
    extractor = RiemannExtractor({
        "views": [_raw_view(weight=2.0), _grouped_view(weight=3.0)],
        "feature_mode": "mdm",
        "mdm_metric": "logeuclid",
        "mdm_centroid_scope": "class",
    })

    actual = extractor.fit_transform(multichannel_data).predict
    raw, grouped = _manual_views(multichannel_data.features)
    raw_centroids = TorchClassCentroids(metric="logeuclid").fit(
        raw, multichannel_data.target
    )
    grouped_centroids = TorchClassCentroids(metric="logeuclid").fit(
        grouped, multichannel_data.target
    )
    raw_distances = TorchMDMDistances(
        raw_centroids.centroids_, metric="logeuclid"
    ).transform(raw)
    grouped_distances = TorchMDMDistances(
        grouped_centroids.centroids_, metric="logeuclid"
    ).transform(grouped)
    expected = torch.sqrt(
        2.0 * raw_distances.square()
        + (3.0 / 2.0) * grouped_distances.square()
    )

    assert np.array_equal(extractor.classes_, np.array([0, 1]))
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-7, atol=1e-8)


def test_multi_view_median_is_one_coupled_product_centroid(multichannel_data):
    """Median coordinates must come from one jointly weighted product problem."""
    extractor = RiemannExtractor({
        "views": [_raw_view(weight=2.0), _grouped_view(weight=3.0)],
        "feature_mode": "tangent",
        "tangent_metric": "euclid",
        "centroid_type": "median",
        "centroid_params": {"median_tol": 1e-8, "median_max_iter": 200},
    })
    extractor.fit(multichannel_data)

    raw, grouped = _manual_views(multichannel_data.features)
    packed = RaggedBlockSPDBatch((raw.matrices, *grouped.matrices))
    expected = TorchSPDCentroid(
        metric="euclid", centroid_type="median", median_tol=1e-8, median_max_iter=200
    ).fit(packed, block_weights=[2.0, 1.5, 1.5]).centroid_.matrices

    torch.testing.assert_close(
        extractor.tangent_product_centroid_["raw"].matrices, expected[0]
    )
    for actual_block, expected_block in zip(
        extractor.tangent_product_centroid_["groups"].matrices, expected[1:]
    ):
        torch.testing.assert_close(actual_block, expected_block)


def test_class_medians_are_coupled_product_centroids(multichannel_data):
    """Each class median must use shared product distances across all views."""
    extractor = RiemannExtractor({
        "views": [_raw_view(weight=2.0), _grouped_view(weight=3.0)],
        "feature_mode": "mdm",
        "mdm_metric": "euclid",
        "centroid_type": "median",
        "centroid_params": {"median_tol": 1e-8, "median_max_iter": 200},
    }).fit(multichannel_data)

    raw, grouped = _manual_views(multichannel_data.features)
    expected = TorchClassCentroids(
        metric="euclid",
        centroid_type="median",
        median_tol=1e-8,
        median_max_iter=200,
    ).fit(
        RaggedBlockSPDBatch((raw.matrices, *grouped.matrices)),
        multichannel_data.target,
        block_weights=[2.0, 1.5, 1.5],
    )

    for actual_class, expected_class in zip(
        extractor.class_product_centroids_, expected.centroids_
    ):
        torch.testing.assert_close(
            actual_class["raw"].matrices, expected_class.matrices[0]
        )
        for actual_block, expected_block in zip(
            actual_class["groups"].matrices, expected_class.matrices[1:]
        ):
            torch.testing.assert_close(actual_block, expected_block)


def test_cospectra_tangent_features_are_normalised_by_frequency_count(multichannel_data):
    """A spectral view must receive total weight one regardless of frequency count."""
    extractor = RiemannExtractor({
        "views": [{
            "name": "spectral",
            "builder": "cospectra",
            "weight": 1.0,
            "shrinkage": 0.1,
            "params": {
                "window": 64,
                "overlap": 0.75,
                "fmin": 1.0,
                "fmax": 20.0,
                "fs": 100.0,
            },
        }],
        "feature_mode": "tangent",
    })

    actual = extractor.fit_transform(multichannel_data).predict
    signals = extractor._prepare_tensor(multichannel_data.features)
    spd = extractor.view_shrinkages_["spectral"].transform(
        extractor.view_builders_["spectral"].transform_spd(signals)
    )
    block_count = extractor.view_block_counts_["spectral"]
    expected = extractor.tangent_spaces_["spectral"].transform(spd) / math.sqrt(block_count)

    assert block_count > 1
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(("feature_mode", "expected_columns"), [
    ("tangent", 10),
    ("mdm", 2),
    ("both", 12),
])
def test_feature_mode_controls_tangent_and_mdm_parts(
    multichannel_data, feature_mode, expected_columns
):
    """The public feature mode must determine the exact output families."""
    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": feature_mode,
    })

    features = extractor.fit_transform(multichannel_data).predict

    assert features.shape == (8, expected_columns)


def test_global_mdm_returns_one_distance(multichannel_data):
    """Global MDM scope must return one product-distance feature."""
    extractor = RiemannExtractor({
        "views": [_raw_view(), _grouped_view()],
        "feature_mode": "mdm",
        "mdm_centroid_scope": "global",
    })

    features = extractor.fit_transform(multichannel_data).predict

    assert features.shape == (8, 1)
    assert extractor.classes_ is None


class CountingCovarianceBuilder(BaseTorchSPDBuilder):
    """Minimal extension builder that records SPD construction calls."""

    fit_calls = 0
    transform_calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset shared counters between lifecycle scenarios."""
        cls.fit_calls = 0
        cls.transform_calls = 0

    def fit(self, X: torch.Tensor, y=None) -> "CountingCovarianceBuilder":
        """Record builder fitting without introducing estimator state."""
        del X, y
        type(self).fit_calls += 1
        return self

    def transform_spd(self, X: torch.Tensor):
        """Return raw covariance matrices while recording the invocation."""
        type(self).transform_calls += 1
        return TorchCovariances().transform_spd(X)


def _counting_extractor(monkeypatch) -> RiemannExtractor:
    """Register and construct a test-only SPD builder."""
    monkeypatch.setitem(
        RiemannExtractor._SPD_BUILDERS, "counting_covariance", CountingCovarianceBuilder
    )
    return RiemannExtractor({
        "views": [{
            "name": "counting",
            "builder": "counting_covariance",
            "weight": 1.0,
            "shrinkage": 0.1,
            "params": {},
        }],
        "feature_mode": "tangent",
    })


def test_fit_lifecycle_reuses_train_features_and_supports_auto_fit(
    multichannel_data, monkeypatch
):
    """All train paths must construct SPD matrices exactly once."""
    CountingCovarianceBuilder.reset()
    direct = _counting_extractor(monkeypatch)
    direct_features = direct.fit_transform(multichannel_data).predict
    assert CountingCovarianceBuilder.transform_calls == 1

    CountingCovarianceBuilder.reset()
    staged = _counting_extractor(monkeypatch)
    staged.fit(multichannel_data)
    staged_features = staged.transform_for_fit(multichannel_data).predict
    assert CountingCovarianceBuilder.transform_calls == 1

    CountingCovarianceBuilder.reset()
    automatic = _counting_extractor(monkeypatch)
    with pytest.warns(UserWarning, match="RiemannExtractor is not fitted"):
        automatic_features = automatic.transform(
            multichannel_data, use_cache=False
        ).predict
    assert CountingCovarianceBuilder.transform_calls == 1

    np.testing.assert_allclose(direct_features, staged_features)
    np.testing.assert_allclose(direct_features, automatic_features)


def test_class_mdm_auto_fit_requires_target(multichannel_data):
    """Auto-fit must not silently replace unavailable class centroids."""
    data_without_target = _input_data(multichannel_data.features, target=None)
    extractor = RiemannExtractor({
        "views": [_raw_view()],
        "feature_mode": "mdm",
        "mdm_centroid_scope": "class",
    })

    with pytest.warns(UserWarning, match="RiemannExtractor is not fitted"):
        with pytest.raises(ValueError, match="Target data is required"):
            extractor.transform(data_without_target, use_cache=False)


def test_regression_uses_tangent_only_and_rejects_explicit_mdm(multichannel_data):
    """Regression must not construct class-distance features."""
    regression_data = _input_data(
        multichannel_data.features,
        np.linspace(0.0, 1.0, len(multichannel_data.idx)),
        task_type=TaskTypesEnum.regression,
    )
    both = RiemannExtractor({"views": [_raw_view()], "feature_mode": "both"})

    features = both.fit_transform(regression_data).predict

    assert features.shape == (8, 10)
    assert both.effective_feature_mode_ == "tangent"
    with pytest.raises(ValueError, match="feature_mode='mdm'"):
        RiemannExtractor({
            "views": [_raw_view()], "feature_mode": "mdm"
        }).fit(regression_data)


def test_product_median_never_materialises_dense_block_diagonal(
    multichannel_data, monkeypatch
):
    """Joint median fitting must retain the structured SPD representation."""
    def reject_dense(self):
        """Fail if product fitting attempts dense block materialisation."""
        raise AssertionError("to_dense must not be called")

    monkeypatch.setattr(RaggedBlockSPDBatch, "to_dense", reject_dense)
    extractor = RiemannExtractor({
        "views": [_raw_view(), _grouped_view()],
        "feature_mode": "tangent",
        "centroid_type": "median",
        "tangent_metric": "euclid",
    })

    features = extractor.fit_transform(multichannel_data).predict

    assert np.all(np.isfinite(features))


@pytest.mark.parametrize(("params", "match"), [
    ({}, "views must contain"),
    ({"views": [_raw_view(), _raw_view()]}, "unique"),
    ({"views": [{**_raw_view(), "weight": 0.0}]}, "strictly positive"),
    ({"views": [{**_raw_view(), "builder": "unknown"}]}, "Unknown SPD builder"),
])
def test_multi_view_configuration_is_validated(params, match):
    """Invalid product layouts must fail during extractor construction."""
    with pytest.raises((TypeError, ValueError), match=match):
        RiemannExtractor(params)

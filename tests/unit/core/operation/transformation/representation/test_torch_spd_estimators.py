import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.estimation import BlockCovariances, CoSpectra, Covariances

from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    TorchBlockCovariances,
    TorchCoSpectra,
    TorchCovariances,
)


def _signals(
    n_samples: int = 5,
    n_channels: int = 6,
    n_times: int = 256,
    seed: int = 42,
) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_samples, n_channels, n_times))


@pytest.mark.parametrize(
    ("estimator", "keyword_arguments"),
    [
        ("corr", {}),
        ("cov", {}),
        ("scm", {}),
        ("lwf", {}),
        ("oas", {}),
        ("mcd", {"random_state": 42}),
        ("hub", {}),
    ],
)
def test_covariances_matches_pyriemann_for_every_supported_estimator(estimator, keyword_arguments):
    signals = _signals(n_channels=3, n_times=64)
    torch_signals = torch.from_numpy(signals)

    actual_estimator = TorchCovariances(estimator=estimator, **keyword_arguments)
    actual = actual_estimator.fit_transform(torch_signals)
    expected = Covariances(estimator=estimator, **keyword_arguments).fit_transform(signals)

    assert actual_estimator.fit(torch_signals) is actual_estimator
    assert actual.device == torch_signals.device
    assert actual.dtype == torch_signals.dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(
    ("estimator", "keyword_arguments"),
    [
        ("corr", {"bias": True}),
        ("cov", {"bias": True}),
        ("scm", {"assume_centered": True}),
        ("lwf", {"assume_centered": True}),
        ("oas", {"assume_centered": True}),
        ("mcd", {"random_state": 42}),
        ("hub", {"q": 0.8}),
    ],
)
def test_covariances_forwards_estimator_keyword_arguments(estimator, keyword_arguments):
    signals = _signals(n_channels=3, n_times=64)

    actual = TorchCovariances(estimator=estimator, **keyword_arguments).fit_transform(
        torch.from_numpy(signals)
    )
    expected = Covariances(estimator=estimator, **keyword_arguments).fit_transform(signals)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("block_size", [2, [3, 2, 1]])
def test_block_covariances_matches_pyriemann_and_preserves_block_structure(block_size):
    signals = _signals()

    actual_estimator = TorchBlockCovariances(block_size=block_size, estimator="scm")
    actual = actual_estimator.fit_transform(torch.from_numpy(signals))
    expected = BlockCovariances(block_size=block_size, estimator="scm").fit_transform(signals)

    assert actual_estimator.fit(torch.from_numpy(signals)) is actual_estimator
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-8, atol=1e-9)


def test_cospectra_matches_pyriemann_and_exposes_frequencies():
    signals = _signals(n_channels=4)
    parameters = {"window": 64, "overlap": 0.5, "fmin": 2.0, "fmax": 20.0, "fs": 128.0}

    actual_estimator = TorchCoSpectra(**parameters)
    actual = actual_estimator.fit_transform(torch.from_numpy(signals))
    expected_estimator = CoSpectra(**parameters)
    expected = expected_estimator.fit_transform(signals)

    assert actual_estimator.fit(torch.from_numpy(signals)) is actual_estimator
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(actual_estimator.freqs_, expected_estimator.freqs_)

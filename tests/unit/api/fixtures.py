import numpy as np
import pandas as pd
import pytest

NUM_SAMPLES = 50
SERIES_LENGTH = 20


@pytest.fixture
def univariate_time_series_np():
    return np.random.randn(NUM_SAMPLES, SERIES_LENGTH)


@pytest.fixture
def univariate_time_series_df():
    return pd.DataFrame(np.random.randn(NUM_SAMPLES, SERIES_LENGTH))


@pytest.fixture
def multivariate_time_series_np():
    return np.random.randn(NUM_SAMPLES, 3, SERIES_LENGTH)


@pytest.fixture
def multivariate_time_series_df():
    return pd.DataFrame(
        np.random.randn(
            NUM_SAMPLES,
            3,
            SERIES_LENGTH).tolist())


@pytest.fixture
def uni_classification_labels_np():
    return np.random.randint(0, 2, size=NUM_SAMPLES)


@pytest.fixture
def multi_classification_labels_np():
    return np.random.randint(0, 3, size=NUM_SAMPLES)


@pytest.fixture
def uni_classification_labels_df():
    return pd.DataFrame(np.random.randint(0, 2, size=NUM_SAMPLES))


@pytest.fixture
def multi_classification_labels_df():
    return pd.DataFrame(np.random.randint(0, 3, size=NUM_SAMPLES))


@pytest.fixture
def regression_target_np():
    return np.random.randn(NUM_SAMPLES)


@pytest.fixture
def regression_target_df():
    return pd.Series(np.random.randn(NUM_SAMPLES))


@pytest.fixture
def regression_multi_target_np():
    return np.random.randn(NUM_SAMPLES, 3)


@pytest.fixture
def regression_multi_target_df():
    return pd.DataFrame(np.random.randn(NUM_SAMPLES, 3))


# Example usage in a test function:
def test_example(
        univariate_time_series_np,
        univariate_time_series_df,
        multivariate_time_series_np,
        multivariate_time_series_df,
        uni_classification_labels_np,
        multi_classification_labels_df,
        regression_target_np,
        regression_target_df):
    # Perform tests using the generated data
    assert len(univariate_time_series_np) == NUM_SAMPLES
    assert len(univariate_time_series_df) == NUM_SAMPLES
    assert len(multivariate_time_series_np) == NUM_SAMPLES
    assert len(multivariate_time_series_df) == NUM_SAMPLES
    assert len(uni_classification_labels_np) == NUM_SAMPLES
    assert len(multi_classification_labels_df) == NUM_SAMPLES
    assert len(regression_target_np) == NUM_SAMPLES
    assert len(regression_target_df) == NUM_SAMPLES

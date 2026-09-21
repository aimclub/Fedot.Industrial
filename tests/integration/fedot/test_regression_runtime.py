import os

import numpy as np
import pandas as pd
import pytest

from fedot_ind.integration.fedot import RuntimeState, create_regression_runtime


PROFILE = os.getenv("FEDOT_INTEGRATION_PROFILE")


@pytest.mark.skipif(PROFILE not in {"legacy", "tensor"}, reason="Select one FEDOT_INTEGRATION_PROFILE.")
def test_profile_regression_runtime_fit_predict_and_close():
    first = np.linspace(0.0, 29.0, 30)
    second = np.square(first + 1) / 10.0
    train = pd.DataFrame(
        {
            "first": first,
            "second": second,
        },
        index=[f"train-{index}" for index in range(30)],
    )
    target = 3 * train["first"].to_numpy() - 2 * train["second"].to_numpy() + 1
    predict = pd.DataFrame(
        {"first": [6.0, 7.0], "second": [2.0, 5.0]},
        index=["future-a", "future-b"],
    )
    train_before = train.copy(deep=True)
    predict_before = predict.copy(deep=True)

    runtime = create_regression_runtime(PROFILE)
    runtime.fit(train, target)
    result = runtime.predict(predict)

    expected = 3 * predict["first"].to_numpy() - 2 * predict["second"].to_numpy() + 1
    np.testing.assert_allclose(np.asarray(result.values).reshape(-1), expected, rtol=1e-6, atol=1e-5)
    assert result.idx.tolist() == ["future-a", "future-b"]
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(predict, predict_before)
    assert runtime.snapshot.state is RuntimeState.FITTED

    runtime.close()
    assert runtime.snapshot.state is RuntimeState.CLOSED

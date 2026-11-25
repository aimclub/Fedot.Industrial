from __future__ import annotations

import numpy as np
import pytest

from benchmark.industrial.datasets.local_io import load_local_supervised_split, load_local_ts_cases


def test_load_local_supervised_split_supports_timestamped_multivariate_ts(tmp_path):
    train_path = tmp_path / "Toy_TRAIN.ts"
    test_path = tmp_path / "Toy_TEST.ts"
    payload = "\n".join(
        [
            "@problemName Toy",
            "@timestamps true",
            "@univariate false",
            "@dimension 2",
            "@equalLength true",
            "@seriesLength 2",
            "@targetlabel true",
            "@data",
            (
                "(2020-01-01 00:00:00,1.0),(2020-01-01 00:10:00,2.0):"
                "(2020-01-01 00:00:00,3.0),(2020-01-01 00:10:00,4.0):5.0"
            ),
        ]
    )
    train_path.write_text(payload, encoding="utf-8")
    test_path.write_text(payload.replace(":5.0", ":6.0"), encoding="utf-8")

    split = load_local_supervised_split("Toy", train_path=train_path, test_path=test_path)

    assert np.array_equal(split.train_features, np.asarray([[1.0, 2.0, 3.0, 4.0]]))
    assert np.array_equal(split.test_features, np.asarray([[1.0, 2.0, 3.0, 4.0]]))
    assert np.array_equal(split.train_target, np.asarray([5.0]))
    assert np.array_equal(split.test_target, np.asarray([6.0]))
    assert split.metadata["dimensions"] == 2
    assert split.metadata["timestamps"] is True
    assert split.metadata["aligned_channel_grid"] is True


@pytest.mark.parametrize("second_axis, aligned", [((0, 1), True), ((1, 2), False)])
def test_timestamped_cases_keep_missing_values_and_channel_grid_when_reading_verified_bytes(
        tmp_path, second_axis, aligned):
    first, last = second_axis
    payload = (
        "@timestamps true\n@dimension 2\n@targetlabel true\n@data\n"
        f"(0,1),(1,?):({first},3),({last},4):5.0\n"
    ).encode("utf-8")

    cases, targets, metadata = load_local_ts_cases(tmp_path / "not-on-disk.ts", content=payload)

    ((first_channel, second_channel),) = cases
    np.testing.assert_equal(first_channel, [1.0, np.nan])
    np.testing.assert_equal(second_channel, [3.0, 4.0])
    np.testing.assert_equal(targets, [5.0])
    assert metadata["uniform_channel_layout"] is True
    assert metadata["aligned_channel_grid"] is aligned
    assert metadata.get("timestamp_cadence") == (("numeric", "1") if aligned else None)

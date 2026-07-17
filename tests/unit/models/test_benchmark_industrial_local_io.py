from __future__ import annotations

import numpy as np

from benchmark.industrial.datasets.local_io import load_local_supervised_split


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

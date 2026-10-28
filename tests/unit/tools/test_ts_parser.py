import numpy as np
import pandas as pd
import pytest

from fedot_ind.tools.ts_parser import TsFileParseError, parse_ts_file


def write_file(tmp_path, data, label='targetlabel true', timestamps=False):
    path = tmp_path / 'tiny.ts'
    path.write_text('@problemname tiny\n@timestamps ' + str(timestamps).lower()
                    + '\n@univariate false\n@' + label + '\n@data\n' + data, encoding='utf-8')
    return path


def test_numeric_targets_missing_values_and_empty_dimension(tmp_path):
    path = write_file(tmp_path, '1,?,3::2.5\n4,5,6:7,8:3.5\n')
    frame, target = parse_ts_file(path, encoding='utf-8')
    np.testing.assert_array_equal(target.astype(float), [2.5, 3.5])
    np.testing.assert_allclose(frame.iloc[0, 0].to_numpy(), [1, np.nan, 3])
    assert frame.shape == (2, 2) and len(frame.iloc[0, 1]) == 0
    joined = parse_ts_file(path, encoding='utf-8', return_separate_X_and_y=False)
    assert joined.columns.tolist() == ['dim_0', 'dim_1', 'class_vals']


def test_class_labels_are_returned_without_regression_header(tmp_path):
    path = write_file(tmp_path, '1,2:a\n3,4:b\n', label='classlabel true a b')
    frame, target = parse_ts_file(path, encoding='utf-8')
    assert frame.shape == (2, 1)
    np.testing.assert_array_equal(target, ['a', 'b'])


def test_float_timestamp_dialect_retained(tmp_path):
    path = write_file(tmp_path, '(1.5,3),(2.5,4):2\n', timestamps=True)
    frame, target = parse_ts_file(path, encoding='utf-8')
    assert isinstance(frame.iloc[0, 0], pd.Series)
    np.testing.assert_array_equal(frame.iloc[0, 0].index, [1.5, 2.5])
    np.testing.assert_array_equal(frame.iloc[0, 0].to_numpy(), [3, 4])
    assert float(target[0]) == 2


@pytest.mark.parametrize('data', ['', '1,2:3\n@timestamps false\n', '1,2:3\n1:2:3\n'])
def test_incomplete_or_inconsistent_files_rejected(tmp_path, data):
    with pytest.raises(TsFileParseError):
        parse_ts_file(write_file(tmp_path, data), encoding='utf-8')

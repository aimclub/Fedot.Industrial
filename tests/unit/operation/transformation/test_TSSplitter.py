import numpy as np
import pandas as pd
import pytest

from fedot_ind.core.operation.transformation.splitter import TSTransformer


def splitter(strategy):
    return TSTransformer(strategy=strategy)


@pytest.fixture
def frequent_splitter():
    return splitter(strategy='frequent')


@pytest.fixture
def unique_splitter():
    return splitter(strategy='unique')


@pytest.fixture
def time_series():
    return np.random.rand(320)


@pytest.fixture
def anomaly_dict():
    return {'anomaly1': [[40, 50], [60, 80]],
            'anomaly2': [[130, 170], [300, 320]]}


def test_transform_for_fit_frequent_binarize(time_series, anomaly_dict, frequent_splitter):
    result = frequent_splitter.transform_for_fit(series=time_series,
                                                 anomaly_dict=anomaly_dict,
                                                 binarize=True)
    assert isinstance(result, tuple)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)
    assert result[0].shape[0] == result[1].shape[0]
    assert 0 in result[1] and 1 in result[1]
    assert np.sum([len(i) for i in result[0]]) <= len(time_series)
    assert np.mean(result[1]) == 0.5


def test_transform_for_fit_frequent_no_binarize(time_series, anomaly_dict, frequent_splitter):
    result = frequent_splitter.transform_for_fit(series=time_series,
                                                 anomaly_dict=anomaly_dict,
                                                 binarize=False)
    assert isinstance(result, tuple)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)
    assert result[0].shape[0] == result[1].shape[0]
    assert 'no_anomaly' in result[1]
    assert np.sum([len(i) for i in result[0]]) <= len(time_series)
    assert np.where(result[1] == 'no_anomaly', 0, 1).mean() == 0.5


# def test_transform_for_fit_unique_binarize(time_series, anomaly_dict, unique_splitter):
#
#     result = unique_splitter.transform_for_fit(series=time_series,
#                                                anomaly_dict=anomaly_dict,
#                                                binarize=True)
#     dataset1 = result[1][0]
#     dataset2 = result[1][1]
#
#     assert isinstance(result, tuple)
#     assert all([a in anomaly_dict.keys() for a in result[0]])
#     assert len(result[0]) == len(result[1])
#
#     for ds in (dataset1, dataset2):
#         assert isinstance(ds[0], pd.DataFrame)
#         assert isinstance(ds[1], np.ndarray)
#         assert ds[0].shape[0] == ds[1].shape[0]
#         assert np.mean(ds[1]) == 0.5
#
#
# def test_transform_for_fit_unique_no_binarize(time_series, anomaly_dict, unique_splitter):
#     result = unique_splitter.transform_for_fit(series=time_series,
#                                                anomaly_dict=anomaly_dict,
#                                                binarize=False)
#     dataset1 = result[1][0]
#     dataset2 = result[1][1]
#
#     assert isinstance(result, tuple)
#     assert all([a in anomaly_dict.keys() for a in result[0]])
#     assert len(result[0]) == len(result[1])
#
#     for ds in (dataset1, dataset2):
#         assert isinstance(ds[0], pd.DataFrame)
#         assert isinstance(ds[1], np.ndarray)
#         assert ds[0].shape[0] == ds[1].shape[0]
#         assert np.where(ds[1] == 'no_anomaly', 0, 1).mean() == 0.5

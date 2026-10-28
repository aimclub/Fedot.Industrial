import json

import numpy as np
import pytest
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from benchmark.industrial.core import MetricRecord, RunStatus, to_plain_data
from benchmark.industrial.experiments.metric_records import load_metric_records
from benchmark.industrial.models.fedot_input import fedot_input_data
from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor


def test_metric_error_preserves_public_exception_identity():
    from benchmark.industrial.errors import BenchmarkConfigurationError
    from benchmark.industrial.forecasting import BenchmarkConfigurationError as PublicError
    from benchmark.industrial.evaluation.forecasting_metrics import compute_forecasting_metric
    assert PublicError is BenchmarkConfigurationError
    with pytest.raises(PublicError, match='Unsupported metric'):
        compute_forecasting_metric('unknown', np.array([1]), np.array([1]), np.array([1]), 1)


@pytest.mark.parametrize('rows', [1, 2, 3])
def test_non_sliding_windows_preserve_every_row(rows):
    extractor = TorchQuantileExtractor(OperationParameters(use_sliding_window=False))
    values = torch.arange(rows * 23, dtype=torch.float64).reshape(rows, 23)
    result = extractor.apply_window_for_stat_feature_torch(
        values, lambda windows: [windows.mean(dim=-1)], window_size=20)
    expected = values[:, :20].reshape(rows, 4, 5).mean(dim=-1).unsqueeze(-1)
    assert torch.equal(result, expected)
    assert TorchQuantileExtractor.apply_window_for_stat_feature_torch is BaseExtractor.apply_window_for_stat_feature_torch


def test_metric_records_preserve_all_fields(tmp_path):
    record = MetricRecord(run_id='run', benchmark='ucr', dataset_name='one', subset='test',
                          series_id='series', model_name='model', metric_name='accuracy',
                          metric_value=.75, status=RunStatus.SUCCESS)
    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps([to_plain_data(record)]), encoding='utf-8')
    assert load_metric_records(str(path)) == [record]
    assert load_metric_records(None) == []


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_fedot_input_preserves_task_indices_features_and_target(task):
    features, target = np.arange(12).reshape(4, 3), np.arange(4)
    data = fedot_input_data(features, target, task_type=task)
    assert data.task.task_type.name == task
    np.testing.assert_array_equal(data.idx, np.arange(4))
    np.testing.assert_array_equal(data.features, features)
    np.testing.assert_array_equal(data.target, target)

from fedot_ind.core.repository.model_repository import FORECASTING_PREPROC
from fedot_ind.core.operation.transformation.data.forecasting_primitives import (
    ExplainedVarianceRankTruncationImplementation,
    SVDDecompositionImplementation,
    read_forecasting_stage_data,
)
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData
import numpy as np
import pytest

pytest.importorskip('fedot')
pytest.importorskip('torch')


def _build_table_input(rows: int = 24, cols: int = 12, horizon: int = 4) -> InputData:
    matrix = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    target = np.arange(rows * horizon, dtype=float).reshape(rows, horizon)
    return InputData(
        idx=np.arange(rows),
        features=matrix,
        target=target,
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon)),
        data_type=DataTypesEnum.table,
    )


def test_forecasting_preprocessing_repository_contains_new_primitives():
    assert 'svd_decomposition' in FORECASTING_PREPROC
    assert 'randomized_svd_decomposition' in FORECASTING_PREPROC
    assert 'tensor_decomposition' in FORECASTING_PREPROC
    assert 'explained_variance_truncation' in FORECASTING_PREPROC
    assert 'statistical_rank_truncation' in FORECASTING_PREPROC
    assert 'expert_rank_truncation' in FORECASTING_PREPROC


def test_svd_decomposition_operation_emits_stage_metadata():
    input_data = _build_table_input()
    operation = SVDDecompositionImplementation(OperationParameters())

    operation.fit(input_data)
    output = operation.transform_for_fit(input_data)
    stage_data = read_forecasting_stage_data(output)

    assert output.data_type is DataTypesEnum.table
    assert output.features.ndim == 2
    assert 'decomposition' in stage_data
    assert 'basis' in stage_data['decomposition']['runtime']


def test_rank_truncation_operation_uses_upstream_decomposition_state():
    input_data = _build_table_input()
    decomposition = SVDDecompositionImplementation(OperationParameters())
    truncation = ExplainedVarianceRankTruncationImplementation(
        OperationParameters(explained_variance=0.9, min_rank=2)
    )

    decomposed = decomposition.transform_for_fit(input_data)
    truncated = truncation.transform_for_fit(decomposed)
    stage_data = read_forecasting_stage_data(truncated)

    assert truncated.features.ndim == 2
    assert 'decomposition' in stage_data
    assert 'rank_truncation' in stage_data
    assert stage_data['rank_truncation']['diagnostics']['selected_rank'] >= 2

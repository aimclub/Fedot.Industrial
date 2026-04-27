from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    DecompositionResult,
    TensorDevicePolicy,
    compute_randomized_svd_decomposition,
    compute_svd_decomposition,
    compute_tensor_decomposition,
    truncate_decomposition_rank,
)


def _ensure_table_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError('Forecasting primitive operations expect a 2D table-like matrix.')
    return matrix


def _extract_stage_container(holder: Any) -> dict[str, Any]:
    if holder is None:
        return {}
    if isinstance(holder, dict):
        return holder
    existing = getattr(holder, 'forecasting_stage_data', None)
    if existing is None:
        existing = {}
        try:
            setattr(holder, 'forecasting_stage_data', existing)
        except Exception:
            return {}
    return existing if isinstance(existing, dict) else {}


def read_forecasting_stage_data(data: InputData | OutputData) -> dict[str, Any]:
    supplementary = getattr(data, 'supplementary_data', None)
    container = _extract_stage_container(supplementary)
    if container:
        return deepcopy(container)
    return deepcopy(getattr(data, '_forecasting_stage_data', {}))


def write_forecasting_stage_data(
        output: OutputData,
        *,
        stage_name: str,
        diagnostics: dict[str, Any],
        runtime: dict[str, Any] | None = None,
) -> OutputData:
    supplementary = getattr(output, 'supplementary_data', None)
    container = _extract_stage_container(supplementary)
    stage_payload = {'diagnostics': diagnostics}
    if runtime is not None:
        stage_payload['runtime'] = runtime
    container[stage_name] = stage_payload
    if supplementary is not None:
        try:
            setattr(supplementary, 'forecasting_stage_data', container)
        except Exception:
            pass
    setattr(output, '_forecasting_stage_data', container)
    return output


class _BaseForecastingPrimitiveImplementation(DataOperationImplementation):
    stage_name: str = 'primitive'

    def fit(self, input_data: InputData):
        del input_data
        return self

    @staticmethod
    def _convert_matrix_output(input_data: InputData, matrix: np.ndarray) -> OutputData:
        output = OutputData(
            idx=input_data.idx,
            features=matrix,
            predict=matrix,
            target=getattr(input_data, 'target', None),
            task=input_data.task,
            data_type=DataTypesEnum.table,
            supplementary_data=input_data.supplementary_data,
        )
        output.features = matrix
        output.predict = matrix
        output.target = getattr(input_data, 'target', None)
        output.data_type = DataTypesEnum.table
        return output


class _BaseSVDDecompositionImplementation(_BaseForecastingPrimitiveImplementation):
    stage_name = 'decomposition'
    decomposition_strategy = 'full'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.unfolding_strategy = self.params.get('unfolding_strategy', 'channels_last')
        self.n_oversamples = int(self.params.get('n_oversamples', 5))
        self.device_policy = TensorDevicePolicy(device=str(self.params.get('device', 'cpu')))

    def _compute_decomposition(self, matrix: np.ndarray) -> DecompositionResult:
        if self.decomposition_strategy == 'randomized':
            return compute_randomized_svd_decomposition(matrix, n_oversamples=self.n_oversamples)
        if self.decomposition_strategy == 'tensor':
            return compute_tensor_decomposition(matrix, unfolding_strategy=self.unfolding_strategy)
        return compute_svd_decomposition(matrix, strategy='full')

    def _transform(self, input_data: InputData) -> OutputData:
        matrix = _ensure_table_matrix(input_data.features)
        decomposition = self._compute_decomposition(matrix)
        projected = decomposition.projected_features.detach().cpu().numpy()
        output = self._convert_matrix_output(input_data, projected)
        diagnostics = decomposition.to_dict()
        runtime = {
            'basis': decomposition.basis.detach().cpu().numpy(),
            'singular_values': decomposition.singular_values.detach().cpu().numpy(),
            'input_shape': decomposition.input_shape,
            'strategy': decomposition.strategy,
        }
        return write_forecasting_stage_data(
            output,
            stage_name=self.stage_name,
            diagnostics=diagnostics,
            runtime=runtime,
        )

    def transform(self, input_data: InputData) -> OutputData:
        return self._transform(input_data)

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        return self._transform(input_data)


class SVDDecompositionImplementation(_BaseSVDDecompositionImplementation):
    decomposition_strategy = 'full'


class RandomizedSVDDecompositionImplementation(_BaseSVDDecompositionImplementation):
    decomposition_strategy = 'randomized'


class TensorDecompositionImplementation(_BaseSVDDecompositionImplementation):
    decomposition_strategy = 'tensor'


class _BaseRankTruncationImplementation(_BaseForecastingPrimitiveImplementation):
    stage_name = 'rank_truncation'
    rank_policy = 'explained_variance'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.rank = self.params.get('rank')
        self.explained_variance = float(self.params.get('explained_variance', 0.95))
        self.min_rank = int(self.params.get('min_rank', 1))

    def _build_decomposition_from_input(self, input_data: InputData) -> DecompositionResult:
        stage_data = read_forecasting_stage_data(input_data)
        matrix = _ensure_table_matrix(input_data.features)
        decomposition_bundle = stage_data.get('decomposition', {})
        runtime = decomposition_bundle.get('runtime', {})
        if runtime:
            basis = np.asarray(runtime.get('basis'), dtype=float)
            singular_values = np.asarray(runtime.get('singular_values'), dtype=float)
            return DecompositionResult(
                projected_features=self._to_tensor(matrix),
                basis=self._to_tensor(basis),
                singular_values=self._to_tensor(singular_values),
                strategy=str(runtime.get('strategy', 'upstream')),
                input_shape=tuple(runtime.get('input_shape', matrix.shape)),
                metadata=dict(decomposition_bundle.get('diagnostics', {})),
            )
        return compute_svd_decomposition(matrix, strategy='fallback')

    @staticmethod
    def _to_tensor(values: np.ndarray):
        import torch

        return torch.as_tensor(values, dtype=torch.float32)

    def _transform(self, input_data: InputData) -> OutputData:
        decomposition = self._build_decomposition_from_input(input_data)
        truncation = truncate_decomposition_rank(
            decomposition,
            rank=self.rank,
            explained_variance=self.explained_variance,
            policy=self.rank_policy,
            expert_rank=self.rank,
            min_rank=self.min_rank,
        )
        projected = truncation.projected_features.detach().cpu().numpy()
        output = self._convert_matrix_output(input_data, projected)
        diagnostics = truncation.to_dict()
        runtime = {
            'basis': truncation.basis.detach().cpu().numpy(),
            'singular_values': truncation.singular_values.detach().cpu().numpy(),
            'selected_rank': truncation.selected_rank,
            'policy': truncation.policy,
        }
        output = write_forecasting_stage_data(
            output,
            stage_name=self.stage_name,
            diagnostics=diagnostics,
            runtime=runtime,
        )
        decomposition_bundle = read_forecasting_stage_data(input_data).get('decomposition')
        if decomposition_bundle:
            output = write_forecasting_stage_data(
                output,
                stage_name='decomposition',
                diagnostics=dict(decomposition_bundle.get('diagnostics', {})),
                runtime=dict(decomposition_bundle.get('runtime', {})),
            )
        return output

    def transform(self, input_data: InputData) -> OutputData:
        return self._transform(input_data)

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        return self._transform(input_data)


class ExplainedVarianceRankTruncationImplementation(_BaseRankTruncationImplementation):
    rank_policy = 'explained_variance'


class StatisticalRankTruncationImplementation(_BaseRankTruncationImplementation):
    rank_policy = 'statistical'


class ExpertRankTruncationImplementation(_BaseRankTruncationImplementation):
    rank_policy = 'expert'

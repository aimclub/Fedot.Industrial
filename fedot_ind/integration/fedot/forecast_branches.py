"""Opt-in FEDOT adapter with explicit forecast merge semantics, no monkeypatches."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode, _combine_parents_tensordata
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.models.ts_forecasting.branching.contracts import (
    ForecastBranchError, ForecastMergeKind, merge_kind)
from .forecast_merge_context import active_forecast_merge


def merge_forecast_branches(outputs, kind):
    """Exact ordered alignment, equal targets and task horizon; never intersect rows."""
    if not isinstance(kind, ForecastMergeKind) or len(outputs) < 2:
        raise ForecastBranchError('A forecast merge requires an explicit kind and at least two branches')
    first = outputs[0]
    if getattr(first.task.task_type, 'value', None) != 'ts_forecasting':
        raise ForecastBranchError('Forecast merge requires the forecasting task')
    # FEDOT also supports the horizon in TensorData without Task.task_params.
    from fedot.core.data.split.temporal import forecast_horizon
    try:
        horizon = forecast_horizon(first)
    except ValueError as error:
        raise ForecastBranchError(str(error)) from error
    if first.idx is None or len(np.unique(first.idx)) != len(first.idx):
        raise ForecastBranchError('Forecast branches need explicit unique origin indices')
    values = []
    for output in outputs:
        if output.task != first.task or not np.array_equal(output.idx, first.idx):
            raise ForecastBranchError('Forecast branches differ in origin order, task or horizon')
        try:
            output_horizon = forecast_horizon(output)
        except ValueError as error:
            raise ForecastBranchError(str(error)) from error
        if output_horizon != horizon:
            raise ForecastBranchError('Forecast branches have different horizons')
        if (output.target is None) != (first.target is None):
            raise ForecastBranchError('Forecast branches disagree about target presence')
        if output.target is not None:
            if (output.target.dtype != first.target.dtype or output.target.device != first.target.device
                    or not torch.equal(output.target, first.target)):
                raise ForecastBranchError('Forecast branches have different target windows')
        value = output.predict if kind is ForecastMergeKind.PREDICTIONS else output.features
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != len(first.idx):
            raise ForecastBranchError('Each branch must have a two-dimensional tensor aligned with its origins')
        if kind is ForecastMergeKind.PREDICTIONS and value.shape[1] != horizon:
            raise ForecastBranchError('A prediction branch must have exactly one full forecast per origin')
        if kind is ForecastMergeKind.FEATURES and output.predict is not None:
            raise ForecastBranchError('A feature merge cannot consume model predictions')
        if value.is_complex() or not torch.isfinite(value).all():
            raise ForecastBranchError('Forecast merge requires finite real values')
        if values and (value.dtype != values[0].dtype or value.device != values[0].device):
            raise ForecastBranchError('Forecast branches have different dtype or device')
        values.append(value)
    if first.target is not None and (first.target.ndim != 2 or first.target.shape != (len(first.idx), horizon)):
        raise ForecastBranchError('Target must preserve one full horizon per origin')
    return replace(first, features=torch.cat(values, dim=1), predict=None, data_type=DataTypesEnum.table,
                   features_names=None, categorical_idx=[], numerical_idx=[], idx_mapping={})


class ForecastMergeNode(PipelineNode):
    def fit(self, *args, **kwargs):
        token = active_forecast_merge.set(None)
        try:
            return super().fit(*args, **kwargs)
        finally:
            active_forecast_merge.reset(token)

    def predict(self, *args, **kwargs):
        token = active_forecast_merge.set(None)
        try:
            return super().predict(*args, **kwargs)
        finally:
            active_forecast_merge.reset(token)

    def _tensordata_from_parents(self, tensor_data, parent_operation, predictions_cache=None, fold_id=None):
        kind = merge_kind(self.operation.metadata.tags)
        if kind is None:
            raise ForecastBranchError('ForecastMergeNode requires an explicit merge declaration')
        outputs = _combine_parents_tensordata(
            self._nodes_from_with_fixed_order(), tensor_data, parent_operation,
            predictions_cache=predictions_cache, fold_id=fold_id)
        merged = merge_forecast_branches(outputs, kind)
        active_forecast_merge.set(kind)
        return merged


class ForecastPipelineAdapter(PipelineAdapter):
    """Select this adapter explicitly for evolutionary forecast branch graphs."""

    @staticmethod
    def _transform_to_pipeline_node(node):
        from fedot_ind.core.optimizer.graph_validation.capabilities import industrial_capability_index
        content = deepcopy(node.content)
        capability = industrial_capability_index().get(content['name'])
        node_type = ForecastMergeNode if capability is not None and merge_kind(capability.tags) else PipelineNode
        return node_type(operation_type=content['name'], content=content)


class ForecastChangeAdvisor(PipelineChangeAdvisor):
    """Allow model parents only at a declared prediction-merge boundary."""

    def propose_parent(self, node, possible_operations):
        from fedot_ind.core.optimizer.graph_validation.capabilities import industrial_capability_index
        candidates = set(super().propose_parent(node, possible_operations))
        capability = industrial_capability_index().get(node.content['name'])
        if capability is not None and merge_kind(capability.tags) is ForecastMergeKind.PREDICTIONS:
            candidates.update(name for name in possible_operations if name in self.models
                              and name != node.content['name'])
        return candidates

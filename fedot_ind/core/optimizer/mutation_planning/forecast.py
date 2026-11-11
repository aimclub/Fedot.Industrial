"""Opt-in forecast branch proposals, based on shared supervised past windows."""

from dataclasses import replace

from fedot_ind.core.optimizer.graph_validation.contracts import GraphNodeSpec, OperationKind
from fedot_ind.core.models.ts_forecasting.branching.contracts import ForecastMergeKind, merge_kind
from .contracts import ForkForecastBranch, MutationRejection, MutationRejectionCode
from .rewrites import _nodes_for_chain


def _reject(message):
    return MutationRejection(MutationRejectionCode.UNSUPPORTED_EDIT, message)


def apply_forecast_fork(spec, edit, raw_input):
    from ._graph import new_node_id, ordinary
    if (spec.task_type != 'ts_forecasting' or raw_input is None or raw_input.data_type != 'ts'
            or not raw_input.has_target):
        return _reject('Forecast branches require a confirmed univariate forecasting input and target')
    nodes = {node.node_id: node for node in spec.nodes}
    child, window = nodes.get(edit.child_id), nodes.get(edit.window_id)
    kind = merge_kind(edit.merge_operation.tags)
    if child is None or window is None or kind is None or not ordinary(edit.merge_operation):
        return _reject('Forecast merge or window boundary is unavailable')
    if ('ts_to_table' not in window.capabilities.tags or window.capabilities.kind is not OperationKind.TRANSFORM
            or window.capabilities.output_data_types != ('tabular',)):
        return _reject('Branches must share an explicit supervised window boundary')
    if any(not ordinary(op) or 'ts_forecasting' not in op.task_types for op in edit.operations):
        return _reject('Forecast branches contain an unsupported operation')
    # All feature transforms are row-local: no global preprocessing of future targets.
    transforms = edit.operations if kind is ForecastMergeKind.FEATURES else edit.operations[:-1]
    if not transforms or any(op.kind is not OperationKind.TRANSFORM
                             or 'forecast_window_representation' not in op.tags for op in transforms):
        return _reject('Forecast forks require window-local feature representations')
    if kind is ForecastMergeKind.PREDICTIONS and edit.operations[-1].kind is not OperationKind.MODEL:
        return _reject('A prediction branch must end in a model')
    existing_kind = merge_kind(child.capabilities.tags)
    if existing_kind is not None and (kind is not existing_kind or child.operation_name != edit.merge_operation.name):
        return _reject('A merge cannot silently switch its input semantics')
    if existing_kind is None:
        if child.capabilities.kind is not OperationKind.MODEL or child.parent_ids != (window.node_id,):
            return _reject('A new merge must wrap a model of the selected window boundary')
        if any(child.node_id in node.parent_ids for node in spec.nodes):
            return _reject('Only a final model may be wrapped')
    else:
        # A common origin boundary is essential even if row counts happen to match.
        def has_window(node_id, visiting):
            if node_id == window.node_id:
                return True
            if node_id in visiting:
                return False
            node = nodes[node_id]
            return bool(node.parent_ids) and all(has_window(parent, visiting | {node_id}) for parent in node.parent_ids)
        if not all(has_window(parent, set()) for parent in child.parent_ids):
            return _reject('Existing branches do not share the selected window origin')
    inserted = _nodes_for_chain(spec, edit.operations, (window.node_id,), f'forecast:{child.node_id}')
    if existing_kind is not None:
        updated = replace(child, parent_ids=child.parent_ids + (inserted[-1].node_id,))
        return replace(spec, nodes=tuple(updated if node is child else node for node in spec.nodes) + inserted)
    merge_id = new_node_id(spec, window.node_id, child.node_id, edit.merge_operation.name)
    incoming = child.parent_ids if kind is ForecastMergeKind.FEATURES else (child.node_id,)
    merge = GraphNodeSpec(merge_id, edit.merge_operation.name, (*incoming, inserted[-1].node_id), edit.merge_operation)
    remaining = spec.nodes
    if kind is ForecastMergeKind.FEATURES:
        remaining = tuple(replace(node, parent_ids=(merge_id,)) if node is child else node for node in remaining)
    return replace(spec, nodes=remaining + inserted + (merge,))


def forecast_fork_proposals(spec, operations):
    if spec.task_type != 'ts_forecasting':
        return
    merges = tuple(op for op in operations if merge_kind(op.tags))
    representations = tuple(op for op in operations if 'forecast_window_representation' in op.tags)
    models = tuple(op for op in operations if op.kind is OperationKind.MODEL and not merge_kind(op.tags)
                   and 'tabular' in op.input_data_types and 'ts_forecasting' in op.task_types)
    for window in spec.nodes:
        if window.capabilities is None or 'ts_to_table' not in window.capabilities.tags:
            continue
        for child in spec.nodes:
            if child.capabilities is None:
                continue
            if child.capabilities.kind is not OperationKind.MODEL and merge_kind(child.capabilities.tags) is None:
                continue
            for representation in representations:
                for merge in merges:
                    if merge_kind(merge.tags) is ForecastMergeKind.FEATURES:
                        yield ForkForecastBranch(child.node_id, window.node_id, (representation,), merge)
                    else:
                        for model in models:
                            yield ForkForecastBranch(child.node_id, window.node_id, (representation, model), merge)

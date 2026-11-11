"""Atomic multi-node edits. The planner validates the final graph, not intermediate states."""
from dataclasses import replace
from itertools import product

from fedot_ind.core.optimizer.graph_validation.contracts import GraphNodeSpec, OperationKind
from .contracts import (MAX_REWRITE_CHAIN, ForkFeatureBranch, MutationRejection, MutationRejectionCode,
                        PruneFeatureBranch, RewriteChain)


def _reject(message, code=MutationRejectionCode.UNSUPPORTED_EDIT):
    return MutationRejection(code, message)


def _transform(capability, task):
    from ._graph import ordinary
    return ordinary(capability) and capability.kind is OperationKind.TRANSFORM and task in capability.task_types


def _nodes_for_chain(spec, operations, parents, salt, tail_id=None):
    from ._graph import new_node_id
    result = []
    for index, operation in enumerate(operations):
        node_id = (tail_id if tail_id is not None and index == len(operations) - 1 else
                   new_node_id(spec, str(parents), salt, f'{index}:{operation.name}'))
        if node_id in {node.node_id for node in result}:
            raise ValueError('Rewrite node identity collision')
        result.append(GraphNodeSpec(node_id, operation.name, parents, operation))
        parents = (node_id,)
    return tuple(result)


def _valid_chain(spec, operations, parents, raw_input):
    nodes = {node.node_id: node for node in spec.nodes}
    if not all(_transform(operation, spec.task_type) for operation in operations):
        return _reject('Only ordinary task-compatible transforms may form a rewrite')
    if not parents:
        if raw_input is None:
            return _reject('A new primary chain requires confirmed raw input', MutationRejectionCode.INPUT_CONTEXT)
        previous_types = {raw_input.data_type}
    else:
        parent = nodes[parents[0]]
        from ._graph import ordinary
        if not ordinary(parent.capabilities):
            return _reject('Source and special boundaries cannot be rewritten')
        previous_types = set(parent.capabilities.output_data_types)
    for operation in operations:
        if previous_types.isdisjoint(operation.input_data_types):
            return _reject('Rewrite chain has incompatible adjacent types', MutationRejectionCode.DATA_TYPE)
        if operation.requires_target and (raw_input is None or not raw_input.has_target):
            return _reject('Rewrite requires a confirmed target', MutationRejectionCode.TARGET_REQUIRED)
        previous_types = set(operation.output_data_types)
    return None


def apply_rewrite(spec, edit, raw_input):
    from ._graph import homogeneous_input_graph, ordinary
    nodes = {node.node_id: node for node in spec.nodes}
    if isinstance(edit, RewriteChain):
        if any(node_id not in nodes for node_id in edit.node_ids):
            return _reject('Source chain is absent')
        old = [nodes[node_id] for node_id in edit.node_ids]
        from ._graph import unary_context_allowed
        if not unary_context_allowed(spec, *(node.capabilities for node in spec.nodes),
                                     *edit.operations, raw_input=raw_input):
            return _reject('Rewrite input alignment is unconfirmed', MutationRejectionCode.INPUT_CONTEXT)
        if any(not _transform(node.capabilities, spec.task_type) for node in old) or len(old[0].parent_ids) > 1:
            return _reject('Only unary ordinary transform segments may be replaced')
        if any(old[-1].node_id in node.parent_ids and not ordinary(node.capabilities)
               for node in spec.nodes if node.node_id not in edit.node_ids):
            return _reject('Special consumer boundaries cannot be rewritten', MutationRejectionCode.PROTECTED_OPERATION)
        for left, right in zip(old, old[1:]):
            consumers = [node.node_id for node in spec.nodes if left.node_id in node.parent_ids]
            if right.parent_ids != (left.node_id,) or consumers != [right.node_id]:
                return _reject('Rewrite would change a shared or non-contiguous segment')
        invalid = _valid_chain(spec, edit.operations, old[0].parent_ids, raw_input)
        if invalid is not None:
            return invalid
        # Keep a mandatory boundary operation and its parameters when only its input chain changes.
        tail_id = old[-1].node_id if old[-1].operation_name == edit.operations[-1].name else None
        inserted = _nodes_for_chain(spec, edit.operations, old[0].parent_ids, old[-1].node_id, tail_id)
        remaining = tuple(replace(node, parent_ids=tuple(
            inserted[-1].node_id if parent == old[-1].node_id else parent for parent in node.parent_ids))
            for node in spec.nodes if node.node_id not in edit.node_ids)
        return replace(spec, nodes=remaining + inserted)
    if spec.task_type not in ('classification', 'regression'):
        return _reject('Parallel feature rewrites are not declared for this task',
                       MutationRejectionCode.TASK_NOT_SUPPORTED)
    if not homogeneous_input_graph(spec, raw_input):
        return _reject('Feature branch edits require confirmed homogeneous input', MutationRejectionCode.INPUT_CONTEXT)
    child = nodes.get(edit.child_id)
    if child is None or not ordinary(child.capabilities) or not child.parent_ids:
        return _reject('Feature merge is absent')
    if any(nodes[parent].capabilities.kind is not OperationKind.TRANSFORM
           or nodes[parent].capabilities.output_data_types != ('tabular',) for parent in child.parent_ids):
        return _reject('Only aligned tabular transform outputs may be merged', MutationRejectionCode.DATA_TYPE)
    if isinstance(edit, ForkFeatureBranch):
        if edit.parent_id is not None and edit.parent_id not in child.parent_ids:
            return _reject('A fork may reuse only an existing incoming boundary')
        parents = () if edit.parent_id is None else (edit.parent_id,)
        invalid = _valid_chain(spec, edit.operations, parents, raw_input)
        if invalid is not None:
            return invalid
        if edit.operations[-1].output_data_types != ('tabular',):
            return _reject('A feature branch must end in a tabular representation', MutationRejectionCode.DATA_TYPE)
        inserted = _nodes_for_chain(spec, edit.operations, parents, f'fork:{child.node_id}')
        updated = replace(child, parent_ids=child.parent_ids + (inserted[-1].node_id,))
        return replace(spec, nodes=tuple(updated if node is child else node for node in spec.nodes) + inserted)
    if isinstance(edit, PruneFeatureBranch):
        if len(child.parent_ids) < 2 or edit.parent_id not in child.parent_ids:
            return _reject('Pruning must retain another incoming branch')
        uses = {node_id: 0 for node_id in nodes}
        for node in spec.nodes:
            for parent in node.parent_ids:
                if not (node is child and parent == edit.parent_id):
                    uses[parent] += 1
        if uses[edit.parent_id]:
            return _reject('The selected branch output is shared')
        removed, pending = set(), [edit.parent_id]
        while pending:
            node_id = pending.pop()
            if node_id in removed:
                continue
            node = nodes[node_id]
            if not _transform(node.capabilities, spec.task_type):
                if node_id == edit.parent_id:
                    return _reject('Model, source and special subgraphs cannot be pruned')
                continue
            removed.add(node_id)
            for parent in node.parent_ids:
                uses[parent] -= 1
                if uses[parent] == 0:
                    pending.append(parent)
        updated = replace(child, parent_ids=tuple(parent for parent in child.parent_ids if parent != edit.parent_id))
        return replace(spec, nodes=tuple(updated if node is child else node for node in spec.nodes
                                         if node.node_id not in removed))
    raise TypeError('Unsupported rewrite')


def rewrite_proposals(spec, operations, offset):
    """Lazy O(K^3) streams, consumed only under the caller's candidate cap."""
    from .planning import _interleave, _rotated
    transforms = tuple(op for op in _rotated(operations, offset) if _transform(op, spec.task_type))
    nodes = {node.node_id: node for node in spec.nodes}

    def chains(output, length):
        segment = [output.node_id]
        for _ in range(MAX_REWRITE_CHAIN):
            for sequence in product(transforms, repeat=length):
                if tuple(nodes[key].operation_name for key in reversed(segment)) != tuple(op.name for op in sequence):
                    yield RewriteChain(tuple(reversed(segment)), sequence)
            parents = nodes[segment[-1]].parent_ids
            if len(parents) != 1 or not _transform(nodes[parents[0]].capabilities, spec.task_type):
                break
            segment.append(parents[0])

    def forks(child, length):
        for sequence in product(transforms, repeat=length):
            for parent in (None, *child.parent_ids):
                yield ForkFeatureBranch(child.node_id, parent, sequence)

    def prunes(child):
        for parent in child.parent_ids:
            yield PruneFeatureBranch(child.node_id, parent)

    lengths = (*range(2, MAX_REWRITE_CHAIN + 1), 1)
    streams = [chains(node, length) for node in spec.nodes if _transform(node.capabilities, spec.task_type)
               for length in lengths]
    streams += [forks(node, length) for node in spec.nodes if node.parent_ids for length in lengths]
    streams += [prunes(node) for node in spec.nodes if len(node.parent_ids) > 1]
    if spec.task_type == 'ts_forecasting':
        from .forecast import forecast_fork_proposals
        streams.insert(0, forecast_fork_proposals(spec, tuple(_rotated(operations, offset))))
    yield from _interleave(streams)

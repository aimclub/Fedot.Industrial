"""Prepare benchmark parameters before assigning candidate identity and RNG."""

from copy import deepcopy
from dataclasses import dataclass
import json

from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


def configure_pipeline(pipeline, parameters):
    """Apply protocol overrides and detach/sort all JSON parameter mappings."""
    for node in pipeline.nodes:
        name = node.operation.operation_type
        current = node.parameters if isinstance(node.parameters, dict) else {}
        node.parameters = json.loads(canonical_json({**current, **parameters.get(name, {})}))
    return pipeline


def pipeline_configuration(pipeline, task):
    """Ordered-root traversal preserves shared nodes and runtime parent ordering.

    FEDOT reads parents in descriptive-ID order. Parameters must already be
    normalized before calling this function; no branch permutations are invented.
    """
    order, identifiers, parents = [], {}, {}
    pending = [pipeline.root_node]
    while pending:
        node = pending.pop()
        key = id(node)
        if key in identifiers:
            continue
        identifiers[key] = len(order)
        order.append(node)
        incoming = node._nodes_from_with_fixed_order() if node.nodes_from else []
        parents[key] = incoming
        pending.extend(reversed(incoming))
    if len(order) != len(pipeline.nodes):
        raise ValueError('Prepared pipeline must be one connected rooted graph')
    return {
        'schema_version': 'industrial_study_candidate@1', 'task': task,
        'nodes': [{'operation': node.operation.operation_type, 'parameters': node.parameters,
                   'parents': [identifiers[id(parent)] for parent in parents[id(node)]]}
                  for node in order],
    }


@dataclass(frozen=True)
class StudyGraphPreparation:
    adapter: object
    parameters_json: str
    task: str

    @classmethod
    def from_parameters(cls, adapter, parameters, task):
        return cls(adapter, canonical_json(parameters), task)

    def __call__(self, graph):
        with industrial_extension_scope():
            pipeline = self.adapter.restore(deepcopy(graph))
            configure_pipeline(pipeline, json.loads(self.parameters_json))
            configuration = canonical_json(pipeline_configuration(pipeline, self.task))
            return PreparedEvaluationGraph(self.adapter.adapt(pipeline), configuration)


def fitted_model_parameters(pipeline):
    """Snapshot sklearn model parameters after fit, without retaining estimators."""
    from sklearn.base import BaseEstimator
    result = []
    for index, node in enumerate(pipeline.nodes):
        facade = node.fitted_operation
        implementation = getattr(facade, 'implementation', facade)
        estimator = getattr(implementation, 'base_model', implementation)
        if isinstance(estimator, BaseEstimator):
            parameters = json.loads(canonical_json(estimator.get_params(deep=False)))
            result.append({'node_index': index, 'operation': node.operation.operation_type,
                           'class': type(estimator).__name__, 'parameters': parameters})
    return result


def selected_evaluation(individual, prepared, calls):
    """Bind a selected graph to its own successful evaluation, not an aliased mapping."""
    context = individual.metadata.get('evaluation_context')
    if (context is None or context.get('candidate_id') != str(individual.uid)
            or context.get('configuration_id') != prepared.configuration_id
            or canonical_json(context.get('configuration')) != prepared.configuration_json):
        raise ValueError('Selected graph has no matching prepared evaluation evidence')
    matches = [call for call in calls if call.get('evaluation_id') == context.get('evaluation_id')]
    if len(matches) != 1:
        raise ValueError('Selected graph must have exactly one recorded evaluation episode')
    call, = matches
    if (not individual.fitness.valid or not call.get('valid') or call.get('fold_count') != 1
            or call.get('candidate_id') != str(individual.uid)
            or call.get('configuration_id') != prepared.configuration_id
            or canonical_json({key: call.get(key) for key in context}) != canonical_json(context)
            or call.get('fitness') != float(individual.fitness.value)):
        raise ValueError('Selected graph fitness differs from its recorded evaluation episode')
    return deepcopy(context)


def completed_objective_calls(calls, entries):
    """Match accepted completions to native calls; valid returns alone are insufficient."""
    by_id = {call['evaluation_id']: call for call in calls}
    if len(by_id) != len(calls) or len({entry['evaluation_id'] for entry in entries}) != len(entries):
        raise ValueError('Evaluation episodes must have unique identifiers')
    result = []
    for entry in entries:
        call = by_id.get(entry['evaluation_id'])
        context = entry.get('evaluation_context')
        if (call is None or not call['valid'] or call.get('fold_count') != 1 or not isinstance(context, dict)
                or entry['candidate_id'] != call.get('candidate_id') or entry['fitness'] != [call.get('fitness')]
                or canonical_json({key: call.get(key) for key in context}) != canonical_json(context)):
            raise ValueError('Completed archive episode differs from its native evaluation evidence')
        result.append(call)
    return result


def selection_summary(selected, calls, generations, *, completed_entries=None):
    """Separate evaluated candidates from those admitted to generation populations."""
    accepted = calls if completed_entries is None else completed_objective_calls(calls, completed_entries)
    valid = {call['evaluation_id']: call for call in accepted if call['valid']}
    if len(valid) != sum(bool(call['valid']) for call in accepted):
        raise ValueError('Successful evaluation episodes must have unique identifiers')
    members = [member for generation in generations if not generation.get('is_final', False)
               for member in generation['members']]
    admitted = {member['evaluation_id'] for member in members}
    if not admitted or not admitted <= valid.keys():
        raise ValueError('Every generation member must have a successful evaluation episode')
    if any(member['candidate_id'] != valid[member['evaluation_id']]['candidate_id']
           or member['fitness'] != valid[member['evaluation_id']]['fitness'] for member in members):
        raise ValueError('Generation membership differs from its evaluation evidence')
    selected_id = selected['evaluation_context']['evaluation_id']
    if selected_id not in valid or selected['fitness'] != valid[selected_id]['fitness']:
        raise ValueError('Selected episode does not match a successful completed evaluation')
    observed = min(call['fitness'] for call in valid.values())
    admitted_minimum = min(valid[key]['fitness'] for key in admitted)
    return {'observed_minimum': observed, 'generation_population_minimum': admitted_minimum,
            'objective_return_minimum': min(call['fitness'] for call in calls if call['valid']),
            'completed_evaluation_count': len(valid),
            'selected_is_observed_minimum': selected['fitness'] == observed,
            'selected_is_generation_minimum': selected['fitness'] == admitted_minimum,
            'selected_entered_generation_population': selected_id in admitted,
            'outside_generation_population': [
                {'evaluation_id': key, 'candidate_id': call['candidate_id'], 'fitness': call['fitness']}
                for key, call in sorted(valid.items()) if key not in admitted]}


def archive_evidence_matches(record):
    """Require a complete archive and independently replay the final selection audit."""
    try:
        calls, entries, selected = record['objective_calls'], record['completed_archive'], record['selection']
        completed = completed_objective_calls(calls, entries)
        valid_ids = {call['evaluation_id'] for call in calls if call['valid']}
        if (len(completed) != record['diagnostics']['evaluations_succeeded']
                or {entry['evaluation_id'] for entry in entries} != valid_ids):
            return False
        audit = selection_summary(selected, calls, record['generations'], completed_entries=entries)
        context = selected['evaluation_context']
        archived = next(entry for entry in entries if entry['evaluation_id'] == context['evaluation_id'])
        prepared = PreparedEvaluationGraph(None, canonical_json(selected['configuration']))
        return (audit['selected_is_observed_minimum']
                and all(selected.get(key) == value for key, value in audit.items())
                and context == record['selected_evaluation']
                and context == archived['evaluation_context']
                and context['candidate_id'] == selected['candidate_id']
                and context['configuration_id'] == selected['configuration_id'] == prepared.configuration_id
                and canonical_json(context['configuration']) == prepared.configuration_json)
    except (KeyError, ValueError, TypeError, StopIteration):
        return False

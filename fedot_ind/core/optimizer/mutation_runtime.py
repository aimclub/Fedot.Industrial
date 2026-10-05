"""FEDOT/GOLEM effect boundary for immutable structural mutation plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import random

from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator

from golem.core.optimisers.graph import OptGraph, OptNode

from fedot_ind.core.optimizer.graph_validation.adapters import project_runtime_graph
from fedot_ind.core.optimizer.graph_validation import (
    GraphSpec, NodePosition, OperationCapabilities, OperationKind,
    capabilities_from_runtime_node, industrial_capability_index, infer_validation_context,
)
from fedot_ind.core.optimizer.configuration import MutationPlanningPolicy
from fedot_ind.core.optimizer.mutation import mutation_name
from fedot_ind.core.optimizer.mutation_planning import (
    AddBranch, AppliedMutation, DropPrimaryTransform, DropTransform, InsertTransform, RawInputSpec, ReplaceOperation,
    MutationKind, MutationLimits, MutationRejection,
    apply_mutation_plan, enumerate_mutation_plans, repair_graph, select_mutation_plan,
)


@dataclass(frozen=True)
class RuntimeOperationCatalog:
    """Enabled operations with node-factory position restrictions applied."""

    capabilities: tuple[OperationCapabilities, ...]
    unavailable: tuple[str, ...] = ()


def load_runtime_operation_catalog(graph_generation_params) -> RuntimeOperationCatalog:
    """Project enabled operations, never introduce operations from a global pool.

    FEDOT's factory exposes its primary/secondary sets through its repository;
    its get_all_available_operations is not implemented in the pinned version.
    """
    factory = graph_generation_params.node_factory
    repository = getattr(factory, "graph_model_repository", None)
    if repository is not None:
        primary = set(repository.get_operations(is_primary=True))
        secondary = set(repository.get_operations(is_primary=False))
    else:
        getter = getattr(factory, "get_all_available_operations", None)
        names = tuple(getter() or ()) if callable(getter) else ()
        primary, secondary = set(names), set(names)
    declared = industrial_capability_index()
    result, unavailable = [], []
    for name in sorted(primary | secondary):
        if not isinstance(name, str):
            unavailable.append(str(name))
            continue
        capability = declared.get(name)
        if capability is None:
            from fedot.core.pipelines.node import PipelineNode
            capability = capabilities_from_runtime_node(PipelineNode(name), capability_index=declared)
        if capability is None or capability.kind is OperationKind.UNKNOWN:
            unavailable.append(name)
            continue
        positions = []
        if name in primary and capability.supports_position(NodePosition.PRIMARY):
            positions.append(NodePosition.PRIMARY)
        if name in secondary and capability.supports_position(NodePosition.SECONDARY):
            positions.append(NodePosition.SECONDARY)
        if capability.kind is OperationKind.MODEL and (
            name in primary or name in secondary
        ) and capability.supports_position(NodePosition.ROOT):
            positions.append(NodePosition.ROOT)
        if positions:
            result.append(replace(capability, allowed_positions=tuple(positions)))
        else:
            unavailable.append(name)
    return RuntimeOperationCatalog(tuple(result), tuple(unavailable))


def install_extension_aware_advisor(graph_generation_params) -> None:
    """Classify enabled Industrial operations without bypassing FEDOT vetoes.

    Only the standard advisor is copied. A user-defined advisor/subclass retains
    its own policy. The node factory must share the updated classification.
    """
    from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor

    previous = graph_generation_params.advisor
    if type(previous) is not PipelineChangeAdvisor:
        return
    task, _ = infer_validation_context(graph_generation_params)
    declared = industrial_capability_index()
    catalog = load_runtime_operation_catalog(graph_generation_params)
    enabled = tuple(capability for capability in catalog.capabilities
                    if capability.name in declared and task in capability.task_types)
    if not enabled:
        return
    advisor = deepcopy(previous)
    advisor.models = list(dict.fromkeys((*previous.models, *(
        capability.name for capability in enabled if capability.kind is OperationKind.MODEL))))
    advisor.data_operations = list(dict.fromkeys((*previous.data_operations, *(
        capability.name for capability in enabled if capability.kind is OperationKind.TRANSFORM))))
    graph_generation_params.advisor = advisor
    factory = graph_generation_params.node_factory
    if getattr(factory, 'advisor', None) is previous:
        factory.advisor = advisor


def project_for_mutation(graph, graph_generation_params) -> GraphSpec:
    """Use stable positional addresses rather than freshly generated runtime UUIDs."""
    task, device = infer_validation_context(graph_generation_params)
    _, spec = project_runtime_graph(
        graph, adapter=graph_generation_params.adapter, task_type=task, requested_device=device)
    identities = {node.node_id: f"node-{index}" for index, node in enumerate(spec.nodes)}
    return replace(spec, graph_id="mutation-input", nodes=tuple(
        replace(node, node_id=identities[node.node_id],
                parent_ids=tuple(identities.get(parent, f"missing:{parent}") for parent in node.parent_ids))
        for node in spec.nodes
    ))


def interpret_graph_spec(original, before: GraphSpec, after: GraphSpec) -> OptGraph:
    """Materialize a pure graph decision, keeping only unchanged node parameters."""
    from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
    from fedot_ind.integration.fedot.extensions.contracts import IndustrialRuntimeInterface

    original_nodes = tuple(original.nodes)
    if len(original_nodes) != len(before.nodes):
        raise ValueError("Runtime graph does not match mutation projection")
    original_by_id = dict(zip(before.node_ids, original_nodes))
    before_by_id = {node.node_id: node for node in before.nodes}
    declarations = {operation.name: operation for operation in load_industrial_extension_catalog().operations}
    nodes = {}
    for specification in after.nodes:
        previous = before_by_id.get(specification.node_id)
        if previous is not None and previous.operation_name == specification.operation_name:
            content = deepcopy(original_by_id[specification.node_id].content)
        else:
            # Parameters and fitted-node metadata cannot transfer to another model.
            content = {"name": specification.operation_name, "params": {}}
            declaration = declarations.get(specification.operation_name)
            if (declaration is not None
                    and declaration.runtime_interface is IndustrialRuntimeInterface.INPUT_DATA
                    and len(declaration.tasks) > 1 and after.task_type in declaration.tasks):
                content["params"]["task_type"] = after.task_type
        nodes[specification.node_id] = OptNode(content=content)
    for specification in after.nodes:
        nodes[specification.node_id].nodes_from = [nodes[parent] for parent in specification.parent_ids]
    return OptGraph(list(nodes.values()))


def advisor_allows_plan(before: GraphSpec, after: GraphSpec, graph, advisor) -> bool:
    """Preserve FEDOT's domain-specific vetoes that capabilities do not describe."""
    old = {node.node_id: node for node in before.nodes}
    new = {node.node_id: node for node in after.nodes}
    runtime = dict(zip(before.node_ids, graph.nodes))
    for node_id, previous in old.items():
        changed = new.get(node_id)
        protected = any(marker in previous.operation_name for marker in ("data_source", "exog_ts", "custom"))
        if protected and changed != previous:
            return False
        if changed is None:
            can_remove = getattr(advisor, "can_be_removed", None)
            if callable(can_remove) and str(can_remove(runtime[node_id])).split(".")[-1] != "node_rewire":
                return False
        elif changed.operation_name != previous.operation_name:
            propose = getattr(advisor, "propose_change", None)
            if callable(propose) and changed.operation_name not in propose(
                    runtime[node_id], [changed.operation_name]):
                return False
    propose_parent = getattr(advisor, "propose_parent", None)
    if callable(propose_parent):
        for node_id, changed in new.items():
            if node_id not in old:
                continue
            for parent_id in changed.parent_ids:
                if parent_id not in old:
                    name = new[parent_id].operation_name
                    if name not in propose_parent(runtime[node_id], [name]):
                        return False
    return True


class MutationExecutionStatus(str, Enum):
    ACCEPTED = "accepted"
    SKIPPED = "skipped"
    NO_ADMISSIBLE_PLAN = "no_admissible_plan"
    REJECTED = "rejected"
    UNSUPPORTED_ACTION = "unsupported_action"


@dataclass(frozen=True)
class MutationExecution:
    status: MutationExecutionStatus
    action: str = ""
    examined: int = 0
    admissible: int = 0
    repair_edits: int = 0
    issue_codes: tuple[str, ...] = ()
    message: str = ""

    def to_record(self):
        return {"schema_version": "industrial_mutation_execution@1", "status": self.status.value,
                "action": self.action, "examined": self.examined, "admissible": self.admissible,
                "repair_edits": self.repair_edits, "issue_codes": list(self.issue_codes), "message": self.message}


_STRUCTURAL_KINDS = {
    "single_change": MutationKind.REPLACE, "simple": MutationKind.REPLACE,
    "single_add": MutationKind.ADD, "growth": MutationKind.ADD,
    "local_growth": MutationKind.ADD, "tree_growth": MutationKind.ADD,
    "single_drop": MutationKind.DROP, "reduce": MutationKind.DROP,
}


class PlannedMutation:
    """Thin interpreter: no random structural retry after a planned proposal.

    Existing parameter/custom callbacks execute once and pass the same verifier.
    RandomAgent weights are masked to actions with an admissible proposal;
    trained agents retain their own choice and receive explicit failed experience.
    """

    def __init__(self, operator, graph_generation_params, policy: MutationPlanningPolicy, observer=None,
                 *, raw_input: RawInputSpec | None = None):
        self.operator = operator
        self.graph_generation_params = graph_generation_params
        self.policy = policy
        self.catalog = load_runtime_operation_catalog(graph_generation_params)
        self.observer = observer
        self.raw_input = raw_input
        self.last_execution = MutationExecution(MutationExecutionStatus.SKIPPED)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "operator"), name)

    def __getstate__(self):
        # The coordinator callback captures an optimizer, never a worker payload.
        return {**self.__dict__, "observer": None}

    def _record(self, execution: MutationExecution):
        self.last_execution = execution
        if self.observer is not None:
            self.observer(execution)

    def _limits(self, *, repair=False):
        return MutationLimits(max_nodes=self.policy.max_nodes,
                              max_depth=self.operator.requirements.max_depth,
                              max_candidates=self.policy.max_candidates,
                              max_arity=self.operator.requirements.max_arity,
                              max_edits=self.policy.max_repair_edits if repair else 1)

    def __call__(self, population):
        parents = [population] if isinstance(population, Individual) else list(population)
        candidates = [self._mutate(parent) for parent in parents]
        results = [candidate for candidate in candidates if candidate is not None]
        return (results[0] if results else []) if len(parents) == 1 else results

    def _mutate(self, parent):
        before = project_for_mutation(parent.graph, self.graph_generation_params)
        repair = repair_graph(before, limits=self._limits(repair=True), raw_input=self.raw_input)
        if isinstance(repair, MutationRejection):
            self._record(MutationExecution(MutationExecutionStatus.REJECTED,
                         issue_codes=repair.report.error_codes if repair.report else (), message=repair.message))
            return None
        prepared = repair.graph
        if repair.edits and not advisor_allows_plan(
                before, prepared, parent.graph, self.graph_generation_params.advisor):
            self._record(MutationExecution(MutationExecutionStatus.REJECTED,
                         examined=repair.examined, message="Advisor rejected input repair"))
            return None
        operations = self.catalog.capabilities
        limits = replace(self._limits(), max_candidates=max(0, self.policy.max_candidates - repair.examined))
        candidates = enumerate_mutation_plans(prepared, operations, limits=limits, raw_input=self.raw_input)
        plans_by_kind = {kind: [] for kind in MutationKind}
        edit_kinds = {ReplaceOperation: MutationKind.REPLACE, InsertTransform: MutationKind.ADD,
                      AddBranch: MutationKind.ADD, DropTransform: MutationKind.DROP,
                      DropPrimaryTransform: MutationKind.DROP}
        for plan in candidates.plans:
            outcome = apply_mutation_plan(prepared, plan, limits=limits, raw_input=self.raw_input)
            if isinstance(outcome, AppliedMutation) and advisor_allows_plan(
                    before, outcome.graph, parent.graph, self.graph_generation_params.advisor):
                plans_by_kind[edit_kinds[type(plan.edits[0])]].append(plan)
        pools = {kind: replace(candidates, plans=tuple(plans)) for kind, plans in plans_by_kind.items()}
        examined = candidates.examined + repair.examined
        actions = tuple(self.operator.agent.available_actions)
        if isinstance(self.operator.agent, RandomAgent):
            probabilities = tuple(self.operator.agent.get_action_probs(parent.graph))
            if len(probabilities) != len(actions):
                raise ValueError("Mutation probabilities must match configured actions")
            viable = []
            for action, weight in zip(actions, probabilities):
                kind = _STRUCTURAL_KINDS.get(mutation_name(action))
                admissible = bool(pools[kind].plans) if kind is not None else callable(action)
                if weight > 0 and admissible:
                    viable.append((action, weight))
            if not viable:
                self._record(MutationExecution(MutationExecutionStatus.NO_ADMISSIBLE_PLAN,
                             examined=examined,
                             message="No configured positive-weight action has an admissible proposal"))
                return None
            action = random.choices([item[0] for item in viable], [item[1] for item in viable], k=1)[0]
        else:
            action = self.operator.agent.choose_action(parent.graph)
        name = mutation_name(action)
        if not self.operator._will_mutation_be_applied(action):
            self._record(MutationExecution(MutationExecutionStatus.SKIPPED, action=name))
            return parent
        plan = None
        if name in _STRUCTURAL_KINDS:
            pool = pools[_STRUCTURAL_KINDS[name]]
            plan = select_mutation_plan(pool, random.getrandbits(64))
            if plan is None:
                self.operator.agent_experience.collect_experience(parent, action, reward=-1.0)
                self._record(MutationExecution(MutationExecutionStatus.NO_ADMISSIBLE_PLAN, action=name,
                                               examined=examined))
                return None
            outcome = apply_mutation_plan(prepared, plan, limits=self._limits(), raw_input=self.raw_input)
            if not isinstance(outcome, AppliedMutation):
                raise RuntimeError("An enumerated mutation plan could not be applied")
            graph = interpret_graph_spec(parent.graph, before, outcome.graph)
            execution = MutationExecution(MutationExecutionStatus.ACCEPTED, action=name,
                                          examined=examined, admissible=len(pool.plans),
                                          repair_edits=len(repair.edits))
        elif callable(action):
            graph = interpret_graph_spec(parent.graph, before, prepared) if repair.edits else deepcopy(parent.graph)
            graph = self.operator._get_mutation_func(action)(
                graph, requirements=self.operator.requirements,
                graph_gen_params=self.graph_generation_params, parameters=self.operator.parameters)
            changed = project_for_mutation(graph, self.graph_generation_params)
            fixed = repair_graph(changed, limits=self._limits(repair=True), raw_input=self.raw_input)
            if isinstance(fixed, MutationRejection):
                self._record(MutationExecution(MutationExecutionStatus.REJECTED, action=name,
                             issue_codes=fixed.report.error_codes if fixed.report else (), message=fixed.message))
                return None
            if fixed.edits:
                if not advisor_allows_plan(changed, fixed.graph, graph, self.graph_generation_params.advisor):
                    self._record(MutationExecution(MutationExecutionStatus.REJECTED, action=name,
                                 examined=fixed.examined, message="Advisor rejected callback repair"))
                    return None
                graph = interpret_graph_spec(graph, changed, fixed.graph)
            execution = MutationExecution(MutationExecutionStatus.ACCEPTED, action=name,
                                          examined=examined, repair_edits=len(repair.edits) + len(fixed.edits))
        else:
            self._record(MutationExecution(MutationExecutionStatus.UNSUPPORTED_ACTION, action=name))
            return None
        verifier = self.graph_generation_params.verifier
        if not verifier(graph):
            report = getattr(verifier, "last_report", None)
            self._record(replace(execution, status=MutationExecutionStatus.REJECTED,
                                 issue_codes=report.error_codes if report is not None else (),
                                 message="Runtime verifier rejected the planned graph"))
            self.operator.agent_experience.collect_experience(parent, action, reward=-1.0)
            return None
        if graph == parent.graph:
            self._record(replace(execution, status=MutationExecutionStatus.REJECTED,
                                 message="Proposal did not change the graph"))
            return None
        self._record(execution)
        metadata = {**self.operator.requirements.static_individual_metadata,
                    "mutation_execution": execution.to_record()}
        if plan is not None:
            metadata["mutation_plan"] = plan.to_record()
            metadata["mutation_input"] = before.to_record()
            metadata["mutation_prepared_input"] = prepared.to_record()
        return Individual(graph, ParentOperator(type_="mutation", operators=name, parent_individuals=parent),
                          metadata=metadata)


def install_planned_mutation(optimizer, policy: MutationPlanningPolicy):
    if not policy.enabled:
        return
    previous = optimizer.mutation
    observer = getattr(optimizer, "_record_mutation_execution", None)
    optimizer.mutation = PlannedMutation(previous, optimizer.graph_generation_params, policy, observer)
    optimizer.operators = [optimizer.mutation if operator is previous else operator for operator in optimizer.operators]
    optimizer.reproducer.mutation = optimizer.mutation
